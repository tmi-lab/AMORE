from collections import defaultdict

class FPNode:
    def __init__(self,item,parent):
        self.item = item
        self.parent = parent
        self.children = {}
        self.count = 0.
        self.path_sids = []
    
    def add_child(self,citem):
        self.children[citem] = self.children.get(citem,FPNode(citem,parent=self))
        self.children[citem].count+=1
        
    
    def add_sid(self,sid):
        self.path_sids.append(sid)
        
        
class FPTree:
    def __init__(self,transactions,min_support=20):
        self.root = {}
        self.build_tree(transactions,min_support)
        
    def build_tree(self,transactions,min_support):
        item_counts = defaultdict(int)
        for iset in transactions:
            for item in iset:
                item_counts[item]+=1
        frequent_items = {item for item, count in item_counts.items() if count >= min_support}
        for item in frequent_items:
            self.root[item] = self.root.get(item,FPNode(item,parent=None))
            
        for sid, iset in enumerate(transactions):
            
            iset.sort(key=lambda x: item_counts[x], reverse=True)
            try:
                cnode = self.root[iset[0]]
            except KeyError:
                continue
            cnode.count+=1
            for i in range(1,len(iset)):
                cnode.add_child(iset[i])
                cnode = cnode.children[iset[i]]
                
            cnode.path_sids.append(sid)
                
                
                
    def get_itemsets(self,min_support=20,max_depth=10):
        # print("generating frequent itemsets",max_depth)
        itemsets = {}
        for rt,rnode in self.root.items():
            path = []
            itemsets = self.traverse_rnode(path,rnode,itemsets,min_support,max_depth=max_depth)
        return itemsets    
            
                
        
    def traverse_rnode(self,path,rnode,itemsets,min_support=20,max_depth=10):
        if rnode.count >= min_support and len(path)<max_depth:
            new_path = path + [rnode.item]
            itemsets[tuple(new_path)]={'cnt':rnode.count,'sids':rnode.path_sids}
            print(new_path,rnode.count)
            for cit, child in rnode.children.items():
                self.traverse_rnode(new_path,child,itemsets,min_support,max_depth=max_depth)

        return itemsets
        
        
    def merge_branch_sids(self,rnode,sids=[]):
        #print('before merge',len(sids))
        sids = rnode.path_sids + sids
        #print('after merge',len(sids))
        for cit, child in rnode.children.items():
            sids = self.merge_branch_sids(child,sids)
        
        return sids
            
                