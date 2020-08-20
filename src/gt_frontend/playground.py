import inspect
import ast

from typing import Type
import gt_frontend.gtscript as gtscript
from gt_frontend.gtscript import Mesh, Field, Local, Edge, Vertex
from gt_frontend.gtscript_to_gtir import GTScriptToGTIR, VarDeclExtractor
from gtc import common
from gt_frontend.frontend import GTScriptCompilationTask

dtype = common.DataType.FLOAT64

#def edge_reduction(
#    mesh: Mesh,
#    edge_field: Field[Edge, dtype],
#    vertex_field: Field[Vertex, dtype]
#):
#    with computation(FORWARD), interval(0, None), location(Edge) as e:
#        edge_field = sum(vertex_field[v] for v in vertices(e))
#        #edge_field = 0.5 * sum(vertex_field[v] for v in vertices(e))
#        #pass
#        #edge_field[e] = 0.5*sum(vertex_field[v] for v in vertices(e))

#def sparse_ex(
#    mesh: Mesh,
#    edge_field: Field[Edge, dtype],
#    sparse_field: Field[Edge, Local[Vertex], dtype]
#):
#    with computation(FORWARD), interval(0, None), location(Edge) as e:
#        edge_field = sum(sparse_field[e, v] for v in vertices(e))

def test_nested(
    mesh: Mesh,
    f_1: Field[Edge, dtype],
    f_2: Field[Vertex, dtype],
    f_3: Field[Edge, dtype]
):
  with computation(FORWARD), interval(0, None):
    with location(Edge) as e:
      f_1 = 1
    with location(Vertex) as v:
      f_2 = 2
  with computation(FORWARD), interval(0, None), location(Edge) as e:
      f_3 = 3

def fvm_nabla_stencil(
    mesh: Mesh,
    S_MXX: Field[Edge, dtype],
    S_MYY: Field[Edge, dtype],
    pp: Field[Vertex, dtype],
    pnabla_MXX: Field[Vertex, dtype],
    pnabla_MYY: Field[Vertex, dtype],
    vol: Field[Vertex, dtype],
    sign: Field[Vertex, Local[Edge], dtype]
):
    with computation(FORWARD), interval(0, None): #, interval(0, -1):
        with location(Edge) as e:
            zavg = 0.5*sum(pp[v] for v in vertices(e))
            zavg = sum(pp[v] for v in vertices(e))
            zavgS_MXX = S_MXX * zavg
            zavgS_MYY = S_MYY * zavg
        with location(Vertex) as v:
            pnabla_MXX = sum(zavgS_MXX[e] * sign[v, e] for e in edges(v))
            pnabla_MYY = sum(zavgS_MYY[e] * sign[v, e] for e in edges(v))
            pnabla_MXX = pnabla_MXX / vol
            pnabla_MYY = pnabla_MYY / vol


GTScriptCompilationTask(fvm_nabla_stencil).compile()
