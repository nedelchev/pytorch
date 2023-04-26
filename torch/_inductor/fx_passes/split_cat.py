import functools
import logging
import operator
from typing import Dict, List, Union

import torch
from torch._dynamo.utils import counters

from ..pattern_matcher import (
    Arg,
    CallFunction,
    CallMethod,
    FailedMatch,
    get_arg_value,
    Match,
    MULTIPLE,
    PatternEntry,
    PatternExpr,
    PatternMatcherPass,
)

log = logging.getLogger(__name__)


def match_node(node, target_names, op="call_function"):
    return node.op == op and node.target in target_names


# Normalize split with split_size into split_with_sizes, so that we only deal with one type of split in
# subsequent optimizations
class NormalizeSplit(PatternEntry):
    def _get_split_args(self, split_node):
        input_kwarg = "tensor"
        split_size_kwarg = "split_size_or_sections"
        dim_kwarg = "dim"
        if split_node.op == "call_method":
            split_size_kwarg = "split_size"
        return (
            get_arg_value(split_node, 0, input_kwarg),
            get_arg_value(split_node, 1, split_size_kwarg),
            get_arg_value(split_node, 2, dim_kwarg),
        )

    def apply(self, match, graph, node):
        split_node = match.nodes[0]
        split_input, split_size, split_dim = self._get_split_args(split_node)
        if split_input is None or split_dim is None or split_size is None:
            log.warning("couldn't find split args")
            return
        if isinstance(split_size, (list, tuple)) and split_node.op == "call_function":
            return
        if "example_value" not in split_node.meta:
            log.warning("example value absent for node: %s", split_node)
            return
        assert isinstance(split_node.meta["example_value"], (list, tuple))
        split_sections = [t.size()[split_dim] for t in split_node.meta["example_value"]]

        if any(isinstance(section, torch.SymInt) for section in split_sections):
            # TODO dynamic_shapes with assume_static_by_default=False fails while AOT Autograd tracing.
            return
        with graph.inserting_after(split_node):
            new_split_node = graph.call_function(
                torch.split, args=(split_input, split_sections, split_dim)
            )
        split_node.replace_all_uses_with(new_split_node)
        new_split_node.meta.update(split_node.meta)
        graph.erase_node(split_node)
        counters["inductor"]["split_cat_norm"] += 1


def get_sorted_split_users(node: torch.fx.Node) -> List[List[torch.fx.Node]]:
    return [users for split_num, users in sorted(get_split_users(node).items())]


def get_split_users(node: torch.fx.Node) -> Dict[int, List[torch.fx.Node]]:
    split_sections = get_arg_value(node, 1, "split_size_or_sections")
    split_num_to_users = {i: set() for i in range(len(split_sections))}
    for user in node.users.keys():
        assert (
            user.target == operator.getitem
        ), f"Split user not a getitem: {user.target}"
        split_num_to_users[get_arg_value(user, 1)].add(user)
    return {split_num: list(users) for split_num, users in split_num_to_users.items()}


class MergeConsecutiveSplitsPattern(PatternExpr):
    fns = [torch.split]

    def match(self, node: torch.fx.Node) -> Union[Match, FailedMatch]:
        split_sections = get_arg_value(node, 1, "split_size_or_sections")
        if not isinstance(split_sections, (list, tuple)):
            return FailedMatch("Non normalized split node")
        next_level_splits = [None] * len(split_sections)
        for split_num, getitem_nodes in sorted(get_split_users(node).items()):
            if not getitem_nodes:
                continue
            if len(getitem_nodes) > 1:
                # TODO handle this case
                return FailedMatch("multiple getitems for same split_num")
            getitem_node = getitem_nodes[0]
            if not match_node(getitem_node, {operator.getitem}):
                # This should ideally never happen. Split user should always be a getitem.
                return FailedMatch("user of split not a getitem")
            splits_found = []
            for getitem_user in getitem_node.users.keys():
                if match_node(getitem_user, {torch.split}) and get_arg_value(
                    getitem_user, 2, "dim"
                ) == get_arg_value(node, 2, "dim"):
                    splits_found.append(getitem_user)
                else:
                    # Something other than a split found. We can't replace split node for this getitem
                    break
            else:
                if len(splits_found) == 1:
                    next_level_splits[split_num] = splits_found[0]
                # TODO Handle multiple identical splits

        if not next_level_splits or not (any(next_level_splits)):
            return FailedMatch("no next level split found")
        m = Match(self)
        m.nodes = [node] + next_level_splits
        return m


class MergeConsecutiveSplits(PatternEntry):
    def apply(self, match: Match, graph: torch.fx.Graph, node: torch.fx.Node):
        first_split = match.nodes[0]
        next_level_splits = match.nodes[1:]
        combined_sections = self.combine_split_sections(first_split, next_level_splits)
        self.run_replacement(graph, first_split, next_level_splits, combined_sections)

    def combine_split_sections(
        self, first_split: torch.fx.Node, next_level_splits: torch.fx.Node
    ):
        first_split_sections = get_arg_value(first_split, 1, "split_size_or_sections")
        other_splits_sections = [
            get_arg_value(nsplit, 1, "split_size_or_sections") if nsplit else None
            for nsplit in next_level_splits
        ]
        assert len(next_level_splits) == len(first_split_sections)

        new_sections = []
        for split_section, other_split_sections in zip(
            first_split_sections, other_splits_sections
        ):
            if not other_split_sections:
                new_sections.append(split_section)
            else:
                new_sections.extend(other_split_sections)
        return new_sections

    def run_replacement(self, graph, first_split, next_level_splits, combined_sections):
        first_split_input = first_split.args[0]
        first_split_dim = get_arg_value(first_split, 2, "dim")
        # TODO handle multiple getitems with the same arg
        first_split_getitems = sorted(get_split_users(first_split).items())

        to_remove = []
        with graph.inserting_after(first_split_input):
            # Add the new split node
            new_split = graph.call_function(
                torch.split,
                args=(first_split_input, combined_sections, first_split_dim),
            )
            new_split.meta.update(first_split.meta)
            new_getitem_num = 0
            for (old_split_num, old_getitems), next_level_split in zip(
                first_split_getitems, next_level_splits
            ):
                if not old_getitems:
                    new_getitem_num += 1
                    continue
                old_getitem = old_getitems[0]
                if next_level_split is None:  # No split to delete
                    with graph.inserting_after(new_split):
                        new_getitem = graph.call_function(
                            operator.getitem, args=(new_split, new_getitem_num)
                        )
                    new_getitem_num += 1
                    new_getitem.meta.update(old_getitem.meta)
                    old_getitem.replace_all_uses_with(new_getitem)
                else:
                    for next_level_getitems in get_sorted_split_users(next_level_split):
                        if not next_level_getitems:
                            new_getitem_num += 1
                            continue
                        next_level_getitem = next_level_getitems[0]
                        with graph.inserting_after(new_split):
                            new_getitem = graph.call_function(
                                operator.getitem, args=(new_split, new_getitem_num)
                            )
                        new_getitem_num += 1
                        new_getitem.meta.update(next_level_getitem.meta)
                        next_level_getitem.replace_all_uses_with(new_getitem)
                        to_remove.append(next_level_getitem)
                    to_remove.append(next_level_split)
                to_remove.append(old_getitem)
        to_remove.append(first_split)

        for node in to_remove:
            graph.erase_node(node)

        counters["inductor"]["consecutive_split_merged"] += 1
        counters["inductor"]["next_level_split_nodes_removed"] += len(
            list(filter(bool, next_level_splits))
        )
        return to_remove


@functools.lru_cache(None)
def _split_cat_init():
    from .pre_grad import pattern_matcher_passes

    # Pass 1: Normalize split cats
    pass_1 = PatternMatcherPass()
    for pattern in [
        CallFunction(torch.split, Arg(), Arg(), Arg(), _users=MULTIPLE),
        CallFunction(torch.split, Arg(), Arg(), dim=Arg(), _users=MULTIPLE),
        CallFunction(
            torch.split, Arg(), split_size_or_sections=Arg(), dim=Arg(), _users=MULTIPLE
        ),
        CallFunction(
            torch.split,
            tensor=Arg(),
            split_size_or_sections=Arg(),
            dim=Arg(),
            _users=MULTIPLE,
        ),
        CallMethod("split", Arg(), Arg(), Arg(), _users=MULTIPLE),
        CallMethod("split", Arg(), Arg(), dim=Arg(), _users=MULTIPLE),
        CallMethod("split", Arg(), split_size=Arg(), dim=Arg(), _users=MULTIPLE),
    ]:
        pattern = NormalizeSplit(pattern=pattern, extra_check=lambda arg: True)
        pattern.register(pass_1)
    pattern_matcher_passes.append(pass_1)

    # Pass 2: Merge consecutive splits
    pass_2 = PatternMatcherPass()
    pattern = MergeConsecutiveSplits(
        pattern=MergeConsecutiveSplitsPattern(), extra_check=lambda arg: True
    )
    pattern.register(pass_2)

    pattern_matcher_passes.append(pass_2)
