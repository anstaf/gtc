import sys
from gtc.unstructured.usid2 import *
from gtc.unstructured import usid2_codegen

input = Computation(
    name='nabla',
    connectivities=['v2e', 'e2v'],
    params=['S_MXX', 'S_MYY', 'pp', 'pnabla_MXX', 'pnabla_MYY', 'vol', 'sign'],
    temporaries=[
        Temporary(name='zavgS_MXX', dtype='double', location_type='edge'),
        Temporary(name='zavgS_MYY', dtype='double', location_type='edge'),
    ],
    kernels=[
        Kernel(
            location_type='edge',
            primary=Composite(
                name='e',
                items=[
                    Sid(name='e2v'),
                    Sid(name='S_MXX'),
                    Sid(name='S_MYY'),
                    Sid(name='zavgS_MXX'),
                    Sid(name='zavgS_MYY')]),
            secondaries=[Composite(name='v', items=[Sid(name='pp')])],
            body=[
                VarDecl(
                    name='zavg',
                    init=BinaryOp(
                        op=common.BinaryOperator.MUL,
                        left=Literal(dtype='double', value='.5'),
                        right=NeighborReduction(
                            op=ReduceOperator.ADD,
                            dtype='double',
                            connectivity='e2v',
                            max_neighbors=2,
                            has_skip_values=False,
                            primary='e',
                            secondary='v',
                            body=FieldAccess(name='pp', location='v'),
                        ),
                    ),
                ),
                Assign(
                    left=FieldAccess(name='zavgS_MXX', location='e'),
                    right=BinaryOp(
                        op=common.BinaryOperator.MUL,
                        left=FieldAccess(name='S_MXX', location='e'),
                        right=VarAccess(name='zavg')),
                ),
                Assign(
                    left=FieldAccess(name='zavgS_MYY', location='e'),
                    right=BinaryOp(
                        op=common.BinaryOperator.MUL,
                        left=FieldAccess(name='S_MYY', location='e'),
                        right=VarAccess(name='zavg')),
                ),
            ],
        ),
        Kernel(
            location_type='vertex',
            primary=Composite(
                name='v',
                items=[
                    Sid(name='v2e'),
                    Sid(name='pnabla_MXX'),
                    Sid(name='pnabla_MYY'),
                    SparseField(name='sign', connectivity='v2e'),
                    Sid(name='vol')
                ]),
            secondaries=[Composite(name='e', items=[Sid(name='zavgS_MXX'), Sid(name='zavgS_MYY')])],
            body=[
                Assign(
                    left=FieldAccess(name='pnabla_MXX', location='v'),
                    right=NeighborReduction(
                        op=ReduceOperator.ADD,
                        dtype='double',
                        connectivity='v2e',
                        max_neighbors=7,
                        has_skip_values=True,
                        primary='v',
                        secondary='e',
                        body=BinaryOp(
                            op=common.BinaryOperator.MUL,
                            left=FieldAccess(name='zavgS_MXX', location='e'),
                            right=FieldAccess(name='sign', location='v'),
                        ),
                    ),
                ),
                Assign(
                    left=FieldAccess(name='pnabla_MYY', location='v'),
                    right=NeighborReduction(
                        op=ReduceOperator.ADD,
                        dtype='double',
                        connectivity='v2e',
                        max_neighbors=7,
                        has_skip_values=True,
                        primary='v',
                        secondary='e',
                        body=BinaryOp(
                            op=common.BinaryOperator.MUL,
                            left=FieldAccess(name='zavgS_MYY', location='e'),
                            right=FieldAccess(name='sign', location='v'),
                        ),
                    ),
                ),
                Assign(
                    left=FieldAccess(name='pnabla_MXX', location='v'),
                    right=BinaryOp(
                        op=common.BinaryOperator.DIV,
                        left=FieldAccess(name='pnabla_MXX', location='v'),
                        right=FieldAccess(name='vol', location='v'),
                    ),
                ),
                Assign(
                    left=FieldAccess(name='pnabla_MYY', location='v'),
                    right=BinaryOp(
                        op=common.BinaryOperator.DIV,
                        left=FieldAccess(name='pnabla_MYY', location='v'),
                        right=FieldAccess(name='vol', location='v'),
                    ),
                ),
            ],
        )
    ])

if __name__ == "__main__":
    gen = usid2_codegen.gpu if len(sys.argv) > 1 and sys.argv[1] == 'gpu' else usid2_codegen.naive
    print(gen(input))
