import sys
from gtc.unstructured import usid2_codegen, usid2
from gtc import common

_usid2_src = usid2.Computation(
    name='nabla',
    connectivities=['v2e', 'e2v'],
    params=['S_MXX', 'S_MYY', 'pp', 'pnabla_MXX', 'pnabla_MYY', 'vol', 'sign'],
    temporaries=[
        usid2.Temporary(name='zavgS_MXX', dtype='double', location_type='edge'),
        usid2.Temporary(name='zavgS_MYY', dtype='double', location_type='edge'),
    ],
    kernels=[
        usid2.Kernel(
            location_type='edge',
            primary=usid2.Composite(
                name='e',
                items=[
                    usid2.Sid(name='e2v'),
                    usid2.Sid(name='S_MXX'),
                    usid2.Sid(name='S_MYY'),
                    usid2.Sid(name='zavgS_MXX'),
                    usid2.Sid(name='zavgS_MYY')]),
            secondaries=[usid2.Composite(name='v', items=[usid2.Sid(name='pp')])],
            body=[
                usid2.VarDecl(
                    name='zavg',
                    init=usid2.BinaryOp(
                        op=common.BinaryOperator.MUL,
                        left=usid2.Literal(dtype='double', value='.5'),
                        right=usid2.NeighborReduction(
                            op=usid2.ReduceOperator.ADD,
                            dtype='double',
                            connectivity='e2v',
                            max_neighbors=2,
                            has_skip_values=False,
                            primary='e',
                            secondary='v',
                            body=usid2.FieldAccess(name='pp', location='v'),
                        ),
                    ),
                ),
                usid2.Assign(
                    left=usid2.FieldAccess(name='zavgS_MXX', location='e'),
                    right=usid2.BinaryOp(
                        op=common.BinaryOperator.MUL,
                        left=usid2.FieldAccess(name='S_MXX', location='e'),
                        right=usid2.VarAccess(name='zavg')),
                ),
                usid2.Assign(
                    left=usid2.FieldAccess(name='zavgS_MYY', location='e'),
                    right=usid2.BinaryOp(
                        op=common.BinaryOperator.MUL,
                        left=usid2.FieldAccess(name='S_MYY', location='e'),
                        right=usid2.VarAccess(name='zavg')),
                ),
            ],
        ),
        usid2.Kernel(
            location_type='vertex',
            primary=usid2.Composite(
                name='v',
                items=[
                    usid2.Sid(name='v2e'),
                    usid2.Sid(name='pnabla_MXX'),
                    usid2.Sid(name='pnabla_MYY'),
                    usid2.SparseField(name='sign', connectivity='v2e'),
                    usid2.Sid(name='vol')
                ]),
            secondaries=[usid2.Composite(
                name='e', items=[usid2.Sid(name='zavgS_MXX'), usid2.Sid(name='zavgS_MYY')])],
            body=[
                usid2.Assign(
                    left=usid2.FieldAccess(name='pnabla_MXX', location='v'),
                    right=usid2.NeighborReduction(
                        op=usid2.ReduceOperator.ADD,
                        dtype='double',
                        connectivity='v2e',
                        max_neighbors=7,
                        has_skip_values=True,
                        primary='v',
                        secondary='e',
                        body=usid2.BinaryOp(
                            op=common.BinaryOperator.MUL,
                            left=usid2.FieldAccess(name='zavgS_MXX', location='e'),
                            right=usid2.FieldAccess(name='sign', location='v'),
                        ),
                    ),
                ),
                usid2.Assign(
                    left=usid2.FieldAccess(name='pnabla_MYY', location='v'),
                    right=usid2.NeighborReduction(
                        op=usid2.ReduceOperator.ADD,
                        dtype='double',
                        connectivity='v2e',
                        max_neighbors=7,
                        has_skip_values=True,
                        primary='v',
                        secondary='e',
                        body=usid2.BinaryOp(
                            op=common.BinaryOperator.MUL,
                            left=usid2.FieldAccess(name='zavgS_MYY', location='e'),
                            right=usid2.FieldAccess(name='sign', location='v'),
                        ),
                    ),
                ),
                usid2.Assign(
                    left=usid2.FieldAccess(name='pnabla_MXX', location='v'),
                    right=usid2.BinaryOp(
                        op=common.BinaryOperator.DIV,
                        left=usid2.FieldAccess(name='pnabla_MXX', location='v'),
                        right=usid2.FieldAccess(name='vol', location='v'),
                    ),
                ),
                usid2.Assign(
                    left=usid2.FieldAccess(name='pnabla_MYY', location='v'),
                    right=usid2.BinaryOp(
                        op=common.BinaryOperator.DIV,
                        left=usid2.FieldAccess(name='pnabla_MYY', location='v'),
                        right=usid2.FieldAccess(name='vol', location='v'),
                    ),
                ),
            ],
        )
    ])

if __name__ == "__main__":
    gen = usid2_codegen.gpu if len(sys.argv) > 1 and sys.argv[1] == 'gpu' else usid2_codegen.naive
    print(gen(_usid2_src))
