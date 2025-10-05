# encoding: utf-8

"""
A Python 3 compatible implementation of the FP-growth algorithm.
Original Author: Eric Naeseth <eric@naeseth.com>
Updated for Python 3 by ChatGPT
"""

from collections import defaultdict, namedtuple
import csv
from optparse import OptionParser

__author__ = 'Eric Naeseth <eric@naeseth.com>'
__copyright__ = 'Copyright © 2009 Eric Naeseth'
__license__ = 'MIT License'


def find_frequent_itemsets(transactions, minimum_support, include_support=False):
    """Find frequent itemsets in the given transactions using FP-growth."""
    items = defaultdict(int)

    # Đếm tần suất xuất hiện của từng item
    for transaction in transactions:
        for item in transaction:
            items[item] += 1

    # Loại bỏ item có tần suất nhỏ hơn ngưỡng
    items = {item: support for item, support in items.items() if support >= minimum_support}

    def clean_transaction(transaction):
        transaction = [v for v in transaction if v in items]
        transaction.sort(key=lambda v: items[v], reverse=True)
        return transaction

    master = FPTree()
    for transaction in map(clean_transaction, transactions):
        master.add(transaction)

    def find_with_suffix(tree, suffix):
        for item, nodes in tree.items():
            support = sum(n.count for n in nodes)
            if support >= minimum_support and item not in suffix:
                found_set = [item] + suffix
                yield (found_set, support) if include_support else found_set

                cond_tree = conditional_tree_from_paths(tree.prefix_paths(item))
                for s in find_with_suffix(cond_tree, found_set):
                    yield s

    for itemset in find_with_suffix(master, []):
        yield itemset


class FPTree:
    """An FP tree."""

    Route = namedtuple('Route', 'head tail')

    def __init__(self):
        self._root = FPNode(self, None, None)
        self._routes = {}

    @property
    def root(self):
        return self._root

    def add(self, transaction):
        point = self._root
        for item in transaction:
            next_point = point.search(item)
            if next_point:
                next_point.increment()
            else:
                next_point = FPNode(self, item)
                point.add(next_point)
                self._update_route(next_point)
            point = next_point

    def _update_route(self, point):
        assert self is point.tree
        try:
            route = self._routes[point.item]
            route.tail.neighbor = point
            self._routes[point.item] = self.Route(route.head, point)
        except KeyError:
            self._routes[point.item] = self.Route(point, point)

    def items(self):
        for item in self._routes.keys():
            yield (item, self.nodes(item))

    def nodes(self, item):
        try:
            node = self._routes[item].head
        except KeyError:
            return
        while node:
            yield node
            node = node.neighbor

    def prefix_paths(self, item):
        def collect_path(node):
            path = []
            while node and not node.root:
                path.append(node)
                node = node.parent
            path.reverse()
            return path

        return (collect_path(node) for node in self.nodes(item))

    def inspect(self):
        print('Tree:')
        self.root.inspect(1)
        print('\nRoutes:')
        for item, nodes in self.items():
            print(f'  {item!r}')
            for node in nodes:
                print(f'    {node!r}')


def conditional_tree_from_paths(paths):
    """Build a conditional FP-tree from the given prefix paths."""
    tree = FPTree()
    condition_item = None
    items = set()

    for path in paths:
        if condition_item is None:
            condition_item = path[-1].item

        point = tree.root
        for node in path:
            next_point = point.search(node.item)
            if not next_point:
                items.add(node.item)
                count = node.count if node.item == condition_item else 0
                next_point = FPNode(tree, node.item, count)
                point.add(next_point)
                tree._update_route(next_point)
            point = next_point

    if condition_item is None:
        return tree

    for path in tree.prefix_paths(condition_item):
        count = path[-1].count
        for node in reversed(path[:-1]):
            node._count += count

    return tree


class FPNode:
    """A node in an FP tree."""

    def __init__(self, tree, item, count=1):
        self._tree = tree
        self._item = item
        self._count = count
        self._parent = None
        self._children = {}
        self._neighbor = None

    def add(self, child):
        if not isinstance(child, FPNode):
            raise TypeError("Can only add other FPNodes as children")

        if child.item not in self._children:
            self._children[child.item] = child
            child.parent = self

    def search(self, item):
        return self._children.get(item)

    @property
    def tree(self):
        return self._tree

    @property
    def item(self):
        return self._item

    @property
    def count(self):
        return self._count

    def increment(self):
        if self._count is None:
            raise ValueError("Root nodes have no associated count.")
        self._count += 1

    @property
    def root(self):
        return self._item is None and self._count is None

    @property
    def leaf(self):
        return len(self._children) == 0

    @property
    def parent(self):
        return self._parent

    @parent.setter
    def parent(self, value):
        if value is not None and not isinstance(value, FPNode):
            raise TypeError("A node must have an FPNode as a parent.")
        if value and value.tree is not self.tree:
            raise ValueError("Cannot have a parent from another tree.")
        self._parent = value

    @property
    def neighbor(self):
        return self._neighbor

    @neighbor.setter
    def neighbor(self, value):
        if value is not None and not isinstance(value, FPNode):
            raise TypeError("A node must have an FPNode as a neighbor.")
        if value and value.tree is not self.tree:
            raise ValueError("Cannot have a neighbor from another tree.")
        self._neighbor = value

    @property
    def children(self):
        return tuple(self._children.values())

    def inspect(self, depth=0):
        print('  ' * depth + repr(self))
        for child in self.children:
            child.inspect(depth + 1)

    def __repr__(self):
        if self.root:
            return f"<{type(self).__name__} (root)>"
        return f"<{type(self).__name__} {self.item!r} ({self.count!r})>"


if __name__ == '__main__':
    p = OptionParser(usage='%prog data_file')
    p.add_option('-s', '--minimum-support', dest='minsup', type='int',
                 help='Minimum itemset support (default: 2)')
    p.add_option('-n', '--numeric', dest='numeric', action='store_true',
                 help='Convert dataset values to numerals (default: false)')
    p.set_defaults(minsup=2, numeric=False)

    options, args = p.parse_args()
    if len(args) < 1:
        p.error('must provide the path to a CSV file to read')

    transactions = []
    with open(args[0], newline='', encoding='utf-8') as database:
        for row in csv.reader(database):
            if options.numeric:
                transaction = [int(item) for item in row]
                transactions.append(transaction)
            else:
                transactions.append(row)

    result = []
    for itemset, support in find_frequent_itemsets(transactions, options.minsup, True):
        result.append((itemset, support))

    result = sorted(result, key=lambda i: i[0])
    for itemset, support in result:
        print(f"{itemset} {support}")
