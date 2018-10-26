# params['enc.point_0.weight'].data = torch.from_numpy(np.array([
#     [[[1]],[[0]],[[0]]],
#     [[[1]],[[0]],[[0]]],
#     [[[1]],[[0]],[[0]]],
#     [[[1]],[[0]],[[0]]],
# ]).astype(np.float32))
# params['enc.point_0.bias'].data = torch.from_numpy(np.array([0,0,0,0]).astype(np.float32))
# params['enc.conv_0.weight'].data = torch.from_numpy(4*np.array([
#     [[[1,-1,-1],[1,1,-1],[1,-1,-1]]],
#     [[[-1,-1,-1],[-1,1,-1],[1,1,1]]],
#     [[[-1,-1,1],[-1,1,1],[-1,-1,1]]],
#     [[[1,1,1],[-1,1,-1],[-1,-1,-1]]]
# ]).astype(np.float32))
# params['enc.conv_0.bias'].data = torch.from_numpy(np.array([-17.001]*4).astype(np.float32))
# params['enc.point_1.weight'].data = torch.from_numpy(np.array([
#     [[[1]],[[1]],[[1]],[[1]]],
#     [[[1]],[[1]],[[1]],[[1]]],
# ]).astype(np.float32))
# params['enc.point_1.bias'].data = torch.from_numpy(np.array([0, -1]).astype(np.float32))
# params['enc.conv_1.weight'].data = torch.from_numpy(np.array([
#     [[[0,1,0],[1,0,1],[0,1,0]]],
#     [[[0,1,0],[1,0,1],[0,1,0]]],
# ]).astype(np.float32))
# params['enc.conv_1.bias'].data = torch.from_numpy(np.array([-3, -3]).astype(np.float32))
# params['enc.feats_to_classes_op.weight'].data = torch.from_numpy(np.array([
#     [[[-1]],[[0]]],
#     [[[0]],[[-1]]],
# ]).astype(np.float32))
# params['enc.feats_to_classes_op.bias'].data = torch.from_numpy(np.array([0, -1000]).astype(np.float32))

# params['dec.obj_1.conv_0.weight'].data = torch.from_numpy(np.reshape(np.eye(9), (9,1,3,3)).astype(np.float32))
# params['dec.obj_1.conv_0.bias'].data = torch.from_numpy(np.zeros(9).astype(np.float32))
# params['dec.obj_1.point_0.weight'].data = torch.from_numpy(np.reshape(np.eye(9), (9,9,1,1)).astype(np.float32))
# params['dec.obj_1.point_0.bias'].data = torch.from_numpy(np.zeros(9).astype(np.float32))
#
# params['dec.obj_1.conv_1.weight'].data = torch.from_numpy(np.array([
#     [[[0,0,0],[0,0,1],[0,1,1]]],
#     [[[0,1,0],[1,1,1],[1,1,1]]],
#     [[[0,0,0],[1,0,0],[1,1,0]]],
#     [[[0,1,1],[1,1,1],[0,1,1]]],
#     [[[1,1,1],[1,1,1],[1,1,1]]],
#     [[[1,1,0],[1,1,1],[1,1,0]]],
#     [[[0,1,1],[0,0,1],[0,0,0]]],
#     [[[1,1,1],[1,1,1],[0,1,0]]],
#     [[[1,1,0],[1,0,0],[0,0,0]]],
# ]).astype(np.float32))
# params['dec.obj_1.conv_1.bias'].data = torch.from_numpy(np.zeros(9).astype(np.float32))
# params['dec.obj_1.point_1.weight'].data = torch.from_numpy(0.3333*np.ones((9,3,1,1)).astype(np.float32))
# params['dec.obj_1.point_1.bias'].data = torch.from_numpy(np.zeros(3).astype(np.float32))
