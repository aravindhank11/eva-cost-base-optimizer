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
from eva.parser.evaql.evaql_parser import evaql_parser
from eva.parser.evaql.evaql_parserVisitor import evaql_parserVisitor
# from eva.parser.table_ref import TableRef
# from eva.parser.types import FileFormatType
from eva.parser.set_constraint_statement import SetConstraintStatement
from eva.utils.optimizer_constraints import FavorType


class SetConstraint(evaql_parserVisitor):
    def visitSetConstraint(self, ctx: evaql_parser.SetConstraintContext):
        min_accuracy = self.visit(ctx.decimalLiteral(0))
        max_deadline = self.visit(ctx.decimalLiteral(1))
        favors = FavorType[self.visit(ctx.constraintName())]
        stmt = SetConstraintStatement(min_accuracy,max_deadline,favors)
        return stmt

    def visitConstraintName(self, ctx: evaql_parser.ConstraintNameContext):
        if ctx.ACCURACY() is not None:
            return "ACCURACY"
        elif ctx.DEADLINE() is not None:
            return "DEADLINE"

