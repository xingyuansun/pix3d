import os
import sys
import math
import argparse
import datetime
import skimage.measure
import numpy as np
from time import time
from os.path import join
import numba
from numpy.matlib import repmat
from scipy.io import loadmat, savemat
from scipy.interpolate import RegularGridInterpolator as rgi
from tqdm import tqdm
from time import localtime, strftime
import math


def interp3(V, xi, yi, zi, fill_value=0):
    x = np.arange(V.shape[0])
    y = np.arange(V.shape[1])
    z = np.arange(V.shape[2])
    interp_func = rgi((x, y, z), V, 'linear', False, fill_value)
    return interp_func(np.array([xi, yi, zi]).T)


def mesh_grid(input_lr, output_size):
    x_min, x_max, y_min, y_max, z_min, z_max = input_lr
    length = max(max(x_max - x_min, y_max - y_min), z_max - z_min)
    center = np.array([x_max - x_min, y_max - y_min, z_max - z_min]) / 2.
    x = np.linspace(center[0] - length / 2, center[0] + length / 2, output_size[0])
    y = np.linspace(center[1] - length / 2, center[1] + length / 2, output_size[1])
    z = np.linspace(center[2] - length / 2, center[2] + length / 2, output_size[2])
    return np.meshgrid(x, y, z)


def thresholding(V, threshold):
    """
    return the original voxel in its bounding box and bounding box coordinates.
    """
    if V.max() < threshold:
        return np.zeros((2,2,2)), 0, 1, 0, 1, 0, 1
    V_bin = (V >= threshold)
    x_sum = np.sum(np.sum(V_bin, axis=2), axis=1)
    y_sum = np.sum(np.sum(V_bin, axis=2), axis=0)
    z_sum = np.sum(np.sum(V_bin, axis=1), axis=0)

    x_min = x_sum.nonzero()[0].min()
    y_min = y_sum.nonzero()[0].min()
    z_min = z_sum.nonzero()[0].min()
    x_max = x_sum.nonzero()[0].max()
    y_max = y_sum.nonzero()[0].max()
    z_max = z_sum.nonzero()[0].max()
    return V[x_min:x_max+1, y_min:y_max+1, z_min:z_max+1], x_min, x_max, y_min, y_max, z_min, z_max


def get_surface_points(V, threshold, num):
    vtx, faces, _, _ = skimage.measure.marching_cubes_lewiner(V, threshold)
    d = 5e-7 / (V.shape[0] ** 2)
    points = _sample_ptcld_vf(vtx, faces, d)
    while len(points) < num:
        d *= 2
        points = _sample_ptcld_vf(vtx, faces, d)
    idx = np.arange(len(points))
    np.random.shuffle(idx)
    idx = idx[:num]
    return points[idx, :]


def _sample_ptcld_vf(veticies, faces, d=10):
    area = 0
    triangle = veticies[faces]
    density = d
    pt_num_total = 0
    ptnum_list = list()

    for tr in triangle:
        tr_a = np.linalg.norm(np.cross(tr[1] - tr[0], tr[2] - tr[0])) / 2
        area += tr_a
        ptnum = max(int(tr_a * density), 1)
        pt_num_total += ptnum
        ptnum_list.append(ptnum)

    point = np.zeros([pt_num_total, 3], np.float32)
    cnt = 0
    for idx_tr, tr in enumerate(triangle):
        ptnum = ptnum_list[idx_tr]
        sqrt_r1 = np.sqrt(np.random.rand(ptnum))
        r2 = np.random.rand(ptnum)
        pts = np.outer(1 - sqrt_r1, tr[0]) + np.outer((sqrt_r1) * (1 - r2), tr[1]) + np.outer(r2 * sqrt_r1, tr[2])
        point[cnt:cnt + ptnum, :] = pts
        cnt += ptnum
    return point


def load_list_from_file(file_path):
    with open(file_path) as f:
        lines = f.read().split('\n')
    while lines.count('') > 0:
        lines.remove('')
    return lines


def load_voxel_and_crop(data_path, crop_len, var_name, max_value):
    """
    load voxel and crop a few outer voxels
    """
    if data_path.endswith('.mat'):
        try:
            voxel = loadmat(data_path)[var_name]
        except TypeError as err:
            import pdb; pdb.set_trace()
    else:
        voxel = np.load(data_path)
    assert voxel.shape[0] == voxel.shape[1] and voxel.shape[0] == voxel.shape[2]
    voxel = np.array(voxel, dtype=float) / max_value
    pred_len = voxel.shape[0]
    voxel = voxel[crop_len:pred_len - crop_len,
                  crop_len:pred_len - crop_len, crop_len:pred_len - crop_len]
    return voxel


def load_pts_from_list(path_list, mode, pts_size, crop_len, threshold, var_name, max_value, use_tqdm=True):
    """
    mode:
        points: load existing points
        interior: sample points from inside voxels
        surface: sample points from voxel isosurface vertices
    All point sets are normalized to min 0, max 1.
    """
    pts_list = []
    print('loading and generating point lists...')
    it = range(len(path_list))
    if use_tqdm:
        it = tqdm(it)
    for cnt in it:
        data_path = path_list[cnt]
        empty_voxel = False
        if mode == 'voxels':
            voxel = load_voxel_and_crop(data_path, crop_len, var_name, max_value)
            if voxel.max() < threshold:
                # dummy isosurface
                empty_voxel = True
                points = np.zeros((pts_size, 3))
            else:
                points = get_surface_points(voxel, threshold, pts_size)
        elif mode == 'points':
            points = np.loadtxt(data_path)
            assert points.shape[0] == pts_size and points.shape[1] == 3

        if not empty_voxel:
            # normalize point clouds: set center of bounding box to origin, longest side to 1
            bound_l = np.min(points, axis=0)
            bound_h = np.max(points, axis=0)
            points = points - (bound_l + bound_h) / 2
            points = points / (bound_h - bound_l).max()

        pts_list.append(points)
    print('finish loading\n')
    return np.array(pts_list)


def load_pts_from_list_parallel(path_list, *args):
    import multiprocessing
    job_args = list()
    nprocesses = 16
    job_size = int(math.ceil(len(path_list) / nprocesses))
    for indj in range(nprocesses):
        indl = job_size * indj
        indh = min(job_size * (indj + 1), len(path_list))
        if indh > indl:
            job_args.append((path_list[indl:indh], *args, True))

    with multiprocessing.Pool(processes=nprocesses) as pool:
        results = pool.starmap(load_pts_from_list, job_args)
    return np.concatenate(results)


downsample_uneven_warned = False


def downsample(vox_in, times, use_max=True):
    global downsample_uneven_warned
    if vox_in.shape[0] % times != 0 and not downsample_uneven_warned:
        print('WARNING: not dividing the space evenly.')
        downsample_uneven_warned = True
    return _downsample(vox_in, times, use_max=use_max)


@numba.jit(nopython=True, cache=True)
def _downsample(vox_in, times, use_max=True):
    dim = vox_in.shape[0] // times
    vox_out = np.zeros((dim, dim, dim))
    for x in range(dim):
        for y in range(dim):
            for z in range(dim):
                subx = x * times
                suby = y * times
                subz = z * times
                subvox = vox_in[subx:subx + times,
                                suby:suby + times, subz:subz + times]
                if use_max:
                    vox_out[x, y, z] = np.max(subvox)
                else:
                    vox_out[x, y, z] = np.mean(subvox)
    return vox_out


def downsample_voxel(voxel, threshold, output_size, resample=True):
    if voxel.shape[0] > 100:
        assert output_size[0] in (32, 128)
        # downsample to 32 before finding bounding box
        if output_size[0] == 32:
            voxel = downsample(voxel, 4, use_max=True)
    if not resample:
        return voxel

    voxel, x_min, x_max, y_min, y_max, z_min, z_max = thresholding(
        voxel, threshold)
    x_mesh, y_mesh, z_mesh = mesh_grid(
        (x_min, x_max, y_min, y_max, z_min, z_max), output_size)
    x_mesh = np.reshape(np.transpose(x_mesh, (1, 0, 2)), (-1))
    y_mesh = np.reshape(np.transpose(y_mesh, (1, 0, 2)), (-1))
    z_mesh = np.reshape(z_mesh, (-1))

    fill_value = 0
    voxel_d = np.reshape(interp3(voxel, x_mesh, y_mesh, z_mesh, fill_value),
                         (output_size[0], output_size[1], output_size[2]))
    return voxel_d


def calc_iou_from_list(path_list1, path_list2, crop_len1, crop_len2, var_name1, var_name2, threshold, rsl, iou_l, iou_r, iou_s, no_resample1, no_resample2, max_value1, max_value2):
    iou = []
    print('loading voxels and calculating iou...')
    for cnt in tqdm(range(len(path_list1))):
        voxel1 = load_voxel_and_crop(path_list1[cnt], crop_len1, var_name1, max_value1)
        voxel1 = downsample_voxel(
            voxel1, threshold, output_size=[rsl, rsl, rsl], resample=not no_resample1)

        voxel2 = load_voxel_and_crop(path_list2[cnt], crop_len2, var_name2, max_value2)
        voxel2 = downsample_voxel(
            voxel2, threshold, output_size=[rsl, rsl, rsl], resample=not no_resample2)

        iou.append([])
        for thres_iou in np.arange(iou_l, iou_r, iou_s):
            voxel1_thres = voxel1 > thres_iou
            voxel2_thres = voxel2 > thres_iou

            iou_score = np.sum(np.logical_and(voxel1_thres, voxel2_thres)) / np.sum(np.logical_or(voxel1_thres, voxel2_thres))
            if np.isnan(iou_score):
                iou_score = 0.
                print("Warning: found empty voxel pairs at threshold %f" % threshold)
            iou[-1].append(iou_score)

    iou = np.array(iou)
    max_iou = -1
    max_iou_thres = 0
    max_iou_cnt = 0
    for cnt, thres_iou in enumerate(np.arange(iou_l, iou_r, iou_s)):
        if np.mean(iou[:, cnt]) > max_iou:
            max_iou = np.mean(iou[:, cnt])
            max_iou_thres = thres_iou
            max_iou_cnt = cnt
    print('finished loading voxels and calculating iou\n')
    return max_iou_thres, iou[:, max_iou_cnt]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # paths
    parser.add_argument('--list1_path', type=str, required=True,
                        help='path to the 1st file list')
    parser.add_argument('--list2_path', type=str, required=True,
                        help='path to the 2nd file list')
    parser.add_argument('--output_path', type=str, required=True,
                        help='path to the output -- a csv file containing the scores')
    # loading
    parser.add_argument('--list1_mode', type=str, default='voxels', choices=('voxels', 'points'),
                        help='type of inputs expected for files from list 1: "voxels" for voxel inputs \
                              (for CD and EMD calculations, point clouds are sampled from the isosurface); "points" for point cloud inputs')
    parser.add_argument('--list2_mode', type=str, default='voxels', choices=('voxels', 'points'),
                        help='type of inputs expected for files from list 2: "voxels" for voxel inputs \
                              (for CD and EMD calculations, point clouds are sampled from the isosurface); "points" for point cloud inputs')
    parser.add_argument('--list1_max_value', type=float, default=1, help="voxel inputs from list 1 will be divided by this value")
    parser.add_argument('--list2_max_value', type=float, default=1, help="voxel inputs from list 2 will be divided by this value")
    parser.add_argument('--list1_var_name', type=str, default='voxel', help="variable name in .mat file (only for voxels)")
    parser.add_argument('--list2_var_name', type=str, default='voxel', help="variable name in .mat file (only for voxels)")
    # processing
    parser.add_argument('--no_resample1', action='store_true', help='do not perform bounding box finding and resampling for voxels in list 1')
    parser.add_argument('--no_resample2', action='store_true', help='do not perform bounding box finding and resampling for voxels in list 2')
    parser.add_argument('--pts_size', type=int, default=1024,
                        help='number of points to use for CD and EMD calculations')
    parser.add_argument('--threshold', type=float, default=0.1,
                        help='threshold for calculating bounding box (IoU) or isosurface (CD & EMD)')

    parser.add_argument('--batch_size_emd', type=int, default=100,
                        help='batch size for EMD computations')

    parser.add_argument('--calc_iou', action='store_true',
                        help='set to true to calculate IoU (N/A for point cloud inputs)')
    parser.add_argument('--calc_cd', action='store_true',
                        help='set to true to calculate CD')
    parser.add_argument('--calc_emd', action='store_true',
                        help='set to true to calculate EMD')

    parser.add_argument('--iou_l', type=float, default=0.01,
                        help='lower bound of IoU threshold search range')
    parser.add_argument('--iou_r', type=float, default=0.5,
                        help='upper bound of IoU threshold search range')
    parser.add_argument('--iou_s', type=float, default=0.01,
                        help='step size of IoU threshold search range')
    parser.add_argument('--iou_rsl', type=int, default=32,
                        help='resolution of downsampled voxels (for IoU)')
    parser.add_argument('--debug', action='store_true',
                        help='debug mode. only run the first 10 points. ')
    args = parser.parse_args()

    print('starting at time %s' % strftime("%Y-%m-%d %H:%M:%S", localtime()))
    # load list file
    list1 = load_list_from_file(args.list1_path)
    list2 = load_list_from_file(args.list2_path)
    assert len(list1) == len(list2)

    assert any((args.calc_cd, args.calc_iou, args.calc_emd)), 'must specify which metric to use (emd, cd, iou)'
    if args.calc_iou:
        assert args.list1_mode != 'points' or args.list2_mode != 'points', 'iou cannot be calculated for two point clouds.'

    assert len(list2) == len(list1)

    if args.debug:
        list1 = list1[:10]
        list2 = list2[:10]

    if args.calc_cd or args.calc_emd:
        from emd.tf_auctionmatch import EMD
        from cd.tf_nndistance import cd_dis

        # load points (for emd & cd)
        pts1 = load_pts_from_list_parallel(list1, args.list1_mode, args.pts_size,
                                           0, args.threshold, args.list1_var_name, args.list1_max_value)
        pts2 = load_pts_from_list_parallel(list2, args.list2_mode, args.pts_size,
                                           0, args.threshold, args.list2_var_name, args.list2_max_value)

        # calc emd & cd
        print('\ncalculating emd & cd')

        if args.calc_emd:
            emd = EMD(args.pts_size, args.batch_size_emd)
            emd_d = emd.emd_dis(pts1, pts2)

        if args.calc_cd:
            cd_d = cd_dis(pts1, pts2, batch_size=100)
        print('finished calculating emd & cd (time: %s)\n' % strftime("%Y-%m-%d %H:%M:%S", localtime()))

        # print cd score
        if args.calc_cd:
            print('Overall cd: %f' % np.array(cd_d).mean())

        # print emd score
        if args.calc_emd:
            print('Overall emd: %f' % np.array(emd_d).mean())

    # calc iou
    if args.calc_iou:
        print('\ncalculating iou')
        iou_thres, iou = calc_iou_from_list(
            list1, list2, 0, 0, args.list1_var_name, args.list2_var_name, args.threshold, args.iou_rsl, args.iou_l, 
            args.iou_r, args.iou_s, args.no_resample1, args.no_resample2, args.list1_max_value, args.list2_max_value)
        print('finished calculating iou (time: %s)\n' % strftime("%Y-%m-%d %H:%M:%S", localtime()))

        # print iou score
        print('Overall iou: %f' % np.array(iou).mean())


    # write output
    with open(args.output_path, 'w') as f:
        metrics = []
        if args.calc_iou:
            metrics.append('iou_' + str(iou_thres))
        if args.calc_cd:
            metrics.append('cd')
        if args.calc_emd:
            metrics.append('emd')
        f.write(','.join(metrics) + '\n')
        for i in range(len(list1)):
            rst = []
            if args.calc_iou:
                rst.append(str(iou[i]))
            if args.calc_cd:
                rst.append(str(cd_d[i]))
            if args.calc_emd:
                rst.append(str(emd_d[i]))
            f.write(','.join(rst) + '\n')

