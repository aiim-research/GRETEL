import random
from src.explainer.future.metaheuristic.local_search.local_search import LocalSearch
import heapq

class LocalSearchMultiSolutions(LocalSearch):
    def check_configuration(self):
        super().check_configuration()


    def init(self):
        super().init()


    def get_approximation(self, actual, best, min_ctf):
        self.logger.info("Initial solution: " + str(actual))
        self.logger.info("Initial solution size: " + str(len(actual)))
        global_best = best
        result = min_ctf
        priority_queue = []
        heapq.heappush(priority_queue, (len(best), best))
        self.k = 0
        self.first = True
        while(len(priority_queue) > 0):
            self.logger.info("queue: " + str(len(priority_queue)))
            self.logger.info("k: " + str(self.k))
            if(self.k > self.max_oracle_calls) :
                 self.logger.info("Oracle calls limit reached")
                 break
            if(len(global_best) <= 1) :
                 break
            found = False
            size, actual = heapq.heappop(priority_queue)
            if(len(priority_queue) == 0 and self.k < self.max_oracle_calls/2):
                heapq.heappush(priority_queue, (size, actual))
            local_best = actual
            self.logger.info("actual local_best ---> " + str(size))
            
            for s in self.edge_remove(actual):
                if(self.cache.contains(s)):
                    continue
                self.cache.add(s)
                found_, inst = self.evaluate(s)
                if(found_):
                    reduced = s
                    if(len(s) < len(global_best)):
                        found = True
                        global_best = s
                        result = inst
                        self.logger.info("============> (-) Found solution with size: " + str(len(global_best)))
                           
            if(found):
                heapq.heappush(priority_queue, (len(reduced), reduced))
                self.logger.info("============> (-) Enqueue size: " + str(len(reduced)))
                continue
                
            half = int(len(actual) / 2)
            reduce = min(half, random.randint(1, half * 4))
            actual = self.reduce_random(local_best, reduce)
            self.logger.info("actual ---> " + str(len(actual)))
            
            while(len(local_best) - len(actual) > 1):
                for s in self.edge_swap(actual):
                    if(self.cache.contains(s)):
                        continue
                    self.cache.add(s)
                    found_, inst = self.evaluate(s)
                    if(found_):
                        heapq.heappush(priority_queue, (len(s), s))
                        self.logger.info("============> (=) Enqueue size: " + str(len(s)))
                        if(len(s) < len(global_best)):
                            found = True
                            global_best = s
                            result = inst
                            self.logger.info("============> (=) Found solution with size: " + str(len(actual)))
                    
                if(found):
                    break

                actual = self.reduce_random(local_best, len(actual))
                self.logger.info("actual ===> " + str(len(actual)))
                
                for s in self.edge_add(actual, local_best):
                    if(self.cache.contains(s)):
                        continue
                    self.cache.add(s)
                    
                    found_, inst = self.evaluate(s)
                    if(found_):
                        heapq.heappush(priority_queue, (len(s), s))
                        self.logger.info("============> (+) Enqueue size: " + str(len(s)))
                        if(len(s) < len(global_best)):
                            found = True
                            global_best = s
                            result = inst
                            self.logger.info("============> (+) Found solution with size: " + str(len(actual)))
                    
                if(found):
                    break
                
                to_expand = int(((len(local_best) - len(actual)) / 2)) + 1
                expand = len(actual) + min(to_expand, random.randint(1, to_expand * 4))
                # self.logger.info("expand: " + str(expand) + ", local_best: " + str(len(local_best)))
                if(expand > len(local_best)): break
                actual = self.reduce_random(local_best, expand)
                self.logger.info("actual +++> " + str(len(actual)))
                
                
                
        if(self.oracle.predict(result) == self.oracle.predict(self.G)):
            self.logger.info("ERROR, returning non ctf ")
            self.logger.info("instance -> " + str(self.oracle.predict(self.G)))
            self.logger.info("result -> " + str(self.oracle.predict(result)))
        return result
    
    
    
            
            
                