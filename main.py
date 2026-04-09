import os
from _1czi_save import P1
from _2region import P2
from _3czi_hist_cut import P3_hist
from _3czi_position_cut import P3
from _4nuclear_recognize import P4
from _5cytoplasm_recognize import P5
from _6Agg_recognize import P6
from _6Finall import P6_corr
from _6Finall_smooth import P6_smooth
from _6Finall_expresion import P6_expression
from _6Finall_coor_corr import P6_coor_correct
from _7FLI import P7
from _8FLII import P8
from _9qupath import P9_qupath
from _10mapcolor import P10
from _11extraction import P11
from _12add import P12
from _13add_count import P13

pmax = 16400
gpu_id = '1'
read_path = '/nfs/data351/xtw/czi'
paths = os.listdir(read_path)
paths.sort()
print(len(paths))
print(paths)

for i in range(len(paths)):
    czi_path = os.path.join(read_path, paths[i])
    if '1567648' not in czi_path:
        continue
    print(czi_path)
    # input('pause')
    root = P1(czi_path, thum=20, threshold=pmax*0.5)  # CD20 CD3
    '''
    P2(root)
    P3(root, czi_path, channel=0, thum=4, thl=0.1, thh=0, percent=True)   # DAPI
    P3_hist(root, czi_path, channel=1, thum=4, threshold=pmax*0.5, edge=50, show_flag=False)  # CD20
    P3_hist(root, czi_path, channel=2, thum=4, threshold=pmax*0.5, edge=50, show_flag=False)  # CD3
    P3_hist(root, czi_path, channel=3, thum=4, threshold=pmax*0.9, edge=50, show_flag=False)  # CD23
    P3_hist(root, czi_path, channel=4, thum=4, threshold=pmax*0.9, edge=50, show_flag=False)  # CD21
    P4(root, gpu_id)
    P5(root)
    P6(root)
    '''
    # P6_corr(root)
    # P6_smooth(root, ksize=(9, 9), open_iter=7, dilate_iter=1)
    # P6_coor_correct(root)
    # P7(root, threshold=500)
    # P8(root, threshold=500)
    # P9_qupath(root)
    # P6_expression(root)
    # P11(root, pm=0.1721)
    P12(root, czi_path, channel=5, thum=4, threshold=pmax * 0.5, show_flag=False)
    P12(root, czi_path, channel=6, thum=4, threshold=pmax * 0.5, show_flag=False)
    P13(root)
    break

