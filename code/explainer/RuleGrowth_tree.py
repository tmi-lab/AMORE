
from .rule_pattern_miner import *

class RuleNode:
    def __init__(self,fid,parent,left_val,right_val,score=0.,support=0.):
        self.fid = fid
        self.parent = parent
        self.children = {}
        self.score = score
        self.support = support
        self.rule = (left_val,right_val)
    
    def add_child_rule(self,child_rule):
        if child_rule['fid'] in self.children:
            return
        cnode = RuleNode(child_rule['fid'],self,child_rule['left'],child_rule['right'],child_rule['score'],child_rule['support'])
        self.children[child_rule['fid']] = cnode
        
    def add_child(self,cnode):
        if cnode.fid in self.children.keys():
            return
        self.children[cnode.fid] = cnode
        
    def add_children_rules(self,children_rules):
        for child_rule in children_rules:
            self.add_child_rule(child_rule)
            
    def add_children(self,children):
        for child in children:
            self.add_child(child)
        
        
        
class RuleTree:
    def __init__(self,min_support=200,min_score=1.):
        #self.root = {}
        print("init rule tree")
        self.min_support = min_support
        self.min_score = min_score
        # for rule in root_rules:
        #     print(rule)
        #     if rule['support'] < min_support or rule['score'] < min_score:
        #         continue
            # rnode = RuleNode(rule['fid'],None,rule['left'],rule['right'],rule['score'],rule['support'])
            # self.root[rule['fid']] = rnode
        self.root = RuleNode(-1,None,0,0,0,0)
        

                
                
    def get_rule_dict(self):
        #all_rules,leaf_rules = {},{}
        #for rid,rnode in self.root.items():
        all_rules,leaf_rules = self.traverse_node(path=(),node=self.root,rules=[],rule_supports=[],all_rules={},leaf_rules={})
        return all_rules,leaf_rules    
            
                
        
    def traverse_node(self,path,node,rules,rule_supports,all_rules={},leaf_rules={}):
        if node.fid == -1:
            if len(node.children) == 0:
                return all_rules,leaf_rules
            for cid, child in node.children.items():
                all_rules,leaf_rules = self.traverse_node(path,child,rules,rule_supports,all_rules,leaf_rules)
        else:
            new_path = path + (node.fid,) 
            all_rules[new_path]={}
            all_rules[new_path]["rules"]=rules + [(node.fid,">=",node.rule[0]),(node.fid,"<=",node.rule[1])]
            all_rules[new_path]["supports"]=rule_supports + [(node.score,node.support)]

            if len(node.children) == 0:
                leaf_rules[new_path] = all_rules[new_path]
            for cid, child in node.children.items():
                all_rules,leaf_rules = self.traverse_node(new_path,child,all_rules[new_path]["rules"],all_rules[new_path]["supports"],all_rules=all_rules,leaf_rules=leaf_rules)
        # print("#####all rules#####")
        # print(all_rules)
        # print("#####leaf rules#####")
        # print(leaf_rules)
        return all_rules,leaf_rules

