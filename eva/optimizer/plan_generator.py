# coding=utf-8
# Copyright 2018-2022 EVA
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from collections import defaultdict
import math
from eva.catalog.catalog_manager import CatalogManager
from eva.expression.comparison_expression import ComparisonExpression
from eva.expression.function_expression import FunctionExpression
from eva.optimizer.cost_model import CostModel
from eva.optimizer.operators import Operator
from eva.optimizer.optimizer_context import OptimizerContext
from eva.optimizer.optimizer_task_stack import OptimizerTaskStack
from eva.optimizer.optimizer_tasks import BottomUpRewrite, OptimizeGroup, TopDownRewrite
from eva.optimizer.property import PropertyType
from eva.optimizer.rules.rules import RulesManager
from eva.planner.predicate_plan import PredicatePlan
from eva.utils.optimizer_constraints import FavorType, UDFOptimizerConstraints
import pdb
import copy
from eva.planner.seq_scan_plan import SeqScanPlan

class NodeDetails:
    def __init__(self, plan, timeTaken):
        self._timeTaken = timeTaken
        self._plan = plan
        self._hasAccuracy = False

    def setAccuracy(self, accuracy):
        self._accuracy = accuracy
        self._hasAccuracy = True

    @property
    def accuracy(self):
        if (self._hasAccuracy):
            return self._accuracy

    @property
    def hasAccuracy(self):
        return self._hasAccuracy
    
    @property
    def timeTaken(self):
        return self._timeTaken

    @property
    def plan(self):
        return self._plan

class Singleton(type):
    def __init__(cls, name, bases, dict):
        super(Singleton, cls).__init__(name, bases, dict)
        cls.instance = None 

    def __call__(cls,*args,**kw):
        if cls.instance is None:
            cls.instance = super(Singleton, cls).__call__(*args, **kw)
        return cls.instance

class Selectivity(object):
    _selectivity = defaultdict(list)
    _store = 5
    
    @classmethod
    def get_selectivity(self, node):
        last_n = self._selectivity.get(node.__hash__(), [1])
        return sum(last_n) / len(last_n)

    @classmethod
    def update_selectivity(self, node, selectivity):
        if node.__hash__() not in self._selectivity or len(self._selectivity[node.__hash__()]) < self._store:
            self._selectivity[node.__hash__()].append(selectivity)
        else:
            self._selectivity[node.__hash__()].pop()
            self._selectivity[node.__hash__()].append(selectivity)

    @classmethod
    def print(self):
        print(self._selectivity)

class PlanGenerator:
    """
    Used for building Physical Plan from Logical Plan.
    """

    def __init__(
        self, rules_manager: RulesManager = None, cost_model: CostModel = None
    ) -> None:
        self.rules_manager = rules_manager or RulesManager()
        self.cost_model = cost_model or CostModel()

    def execute_task_stack(self, task_stack: OptimizerTaskStack):
        while not task_stack.empty():
            task = task_stack.pop()
            task.execute()

    def build_optimal_physical_plan(
        self, root_grp_id: int, optimizer_context: OptimizerContext
    ):
        physical_plan = None
        root_grp = optimizer_context.memo.groups[root_grp_id]
        best_grp_expr = root_grp.get_best_expr(PropertyType.DEFAULT)
        physical_plan = best_grp_expr.opr

        for child_grp_id in best_grp_expr.children:
            child_plan = self.build_optimal_physical_plan(
                child_grp_id, optimizer_context
            )
            physical_plan.append_child(child_plan)

        return physical_plan

    def filteredUDFs(self, node, targetAccuracy, timeLeft, cardinality):
        if "has_udf" not in node.__dict__:
            return [NodeDetails(node, 0)]

        catalog_manager = CatalogManager()
        possibleUDFs = []
        print(node.__dict__)
        if (isinstance(node, SeqScanPlan)):
            counter = 0
            for expr in node.columns:
                if isinstance(expr, FunctionExpression):
                    type_ = catalog_manager.get_udf_by_name(expr._name).type
                    list_of_name_time_func = catalog_manager.get_udf_with_accuracy_and_time(type_, targetAccuracy, timeLeft, cardinality)
                    for (name, time, accuracy, func) in list_of_name_time_func:
                        newNode = copy.deepcopy(node)
                        newNode.columns[counter].function = func
                        newNode.columns[counter].set_name(name)
                        newNode.clear_children()
                        nodeDetails = NodeDetails(newNode, time)
                        nodeDetails.setAccuracy(accuracy)
                        possibleUDFs.append(nodeDetails)
                counter += 1

        if (isinstance(node, PredicatePlan)):
            if isinstance(node.predicate, ComparisonExpression):
                # pdb.set_trace()
                type_ = catalog_manager.get_udf_by_name(node.predicate.children[0]._name).type
                list_of_name_time_func = catalog_manager.get_udf_with_accuracy_and_time(type_, targetAccuracy, timeLeft, cardinality)
                for (name, time, accuracy, func) in list_of_name_time_func:
                    newNode = copy.deepcopy(node)
                    newNode.predicate.children[0].function = func
                    newNode.predicate.children[0].set_name(name)
                    newNode.clear_children()
                    nodeDetails = NodeDetails(newNode, time)
                    nodeDetails.setAccuracy(accuracy)
                    possibleUDFs.append(nodeDetails)

        return possibleUDFs

    def generatePossiblePlans(self, optimal_plan, targetAccuracy: float, targetTime: float, cardinality: int, plansSoFar=[[0, [], []]]):
        print("In generatePossiblePlans for %s" %optimal_plan)
        if (isinstance(optimal_plan, SeqScanPlan)):
            print("    optimal_plan.__dict__=" %optimal_plan.__dict__)
        if (len(optimal_plan.children) != 0):
            assert (len(optimal_plan.children) == 1) # Works for only 1 children as of now
            plansSoFar = self.generatePossiblePlans(optimal_plan.children[0], targetAccuracy, targetTime,
                        int(cardinality * Selectivity.get_selectivity(optimal_plan)), plansSoFar)
            selectivity = Selectivity.get_selectivity(optimal_plan.children[0])
        else:
            selectivity = 1

        cardinality = int(cardinality * selectivity)

        possiblePlans = []
        atLeastOnePlanForThisNode = False
        for planInfo in plansSoFar:
            timeSpentSoFar, accuracy, plan = planInfo
            for nodeDetails in self.filteredUDFs(optimal_plan, targetAccuracy, targetTime - timeSpentSoFar, cardinality):
                if nodeDetails.hasAccuracy:
                    possiblePlans.append([timeSpentSoFar + nodeDetails.timeTaken, [nodeDetails.accuracy] + accuracy, [nodeDetails.plan] + plan])
                else:
                    possiblePlans.append([timeSpentSoFar + nodeDetails.timeTaken, accuracy, [nodeDetails.plan] + plan])
                atLeastOnePlanForThisNode = True

        # Make sure we have at least 1 viable plan for this node
        if not(atLeastOnePlanForThisNode):
            assert(False)
        else:
            return possiblePlans

    def print(self, plan):
        temp = plan
        while (True):
            print(temp)
            if (temp.children):
                temp = temp.children[0]
            else:
                break

    def optimize(self, logical_plan: Operator):
        optimizer_context = OptimizerContext(self.cost_model)
        memo = optimizer_context.memo
        grp_expr = optimizer_context.add_opr_to_group(opr=logical_plan)
        root_grp_id = grp_expr.group_id
        root_expr = memo.groups[root_grp_id].logical_exprs[0]

        # TopDown Rewrite
        optimizer_context.task_stack.push(
            TopDownRewrite(
                root_expr, self.rules_manager.rewrite_rules, optimizer_context
            )
        )
        self.execute_task_stack(optimizer_context.task_stack)

        # BottomUp Rewrite
        root_expr = memo.groups[root_grp_id].logical_exprs[0]
        optimizer_context.task_stack.push(
            BottomUpRewrite(
                root_expr, self.rules_manager.rewrite_rules, optimizer_context
            )
        )
        self.execute_task_stack(optimizer_context.task_stack)

        # Optimize Expression (logical -> physical transformation)
        root_group = memo.get_group_by_id(root_grp_id)
        optimizer_context.task_stack.push(OptimizeGroup(root_group, optimizer_context))
        self.execute_task_stack(optimizer_context.task_stack)

        # Build Optimal Tree
        optimal_plan = self.build_optimal_physical_plan(root_grp_id, optimizer_context)
        self.print(optimal_plan)

        # Plan UDF selection
        cardinality = 252
        UDFOptimizerConstraints.print()
        print("ACCURACY=%s" %(FavorType.ACCURACY))

        possiblePlans = self.generatePossiblePlans(optimal_plan, UDFOptimizerConstraints.get_min_accuracy(), UDFOptimizerConstraints.get_max_deadline(), cardinality)
        for plan in possiblePlans:
            if (plan[1]):
                plan[1] = sum(plan[1]) / len(plan[1])
            else:
                plan[1] = 0
        
        if UDFOptimizerConstraints.get_favors() == FavorType.ACCURACY:
            possiblePlans.sort(key = lambda t : t[1], reverse=True)
        else:
            possiblePlans.sort(key = lambda t : t[0])
        print("Plans available = %s \n\n" %(possiblePlans))

        time, accuracy, plans = possiblePlans[0]
        udf_optimal_plan = plans[0]
        prev_node = None
        for node in plans:
            if prev_node is not None:
                prev_node.append_child(node)
            prev_node = node

        return udf_optimal_plan

    def build(self, logical_plan: Operator):
        # apply optimizations

        plan = self.optimize(logical_plan)
        return plan
