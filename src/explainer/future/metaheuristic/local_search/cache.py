from collections import OrderedDict

class FixedSizeCache:
    def __init__(self, capacity):
        self.capacity = capacity
        self.cache = OrderedDict()

    def add(self, item):
        """
        Adds a set of integers to the cache.
        If the cache is full, it evicts the oldest item.
        """
        # Convert the set to a frozenset, which is hashable and can be used as a key
        item_key = frozenset(item)
        if item_key in self.cache:
            # If item is already in cache, remove it and re-insert it to mark it as most recently used
            self.cache.move_to_end(item_key)
        else:
            # Add the new item
            self.cache[item_key] = None
            if len(self.cache) > self.capacity:
                # Evict the oldest item
                self.cache.popitem(last=False)

    def contains(self, item):
        """
        Checks if the set of integers is in the cache.
        """
        return frozenset(item) in self.cache