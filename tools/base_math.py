import pickle

import numpy as np

EPS = 1e-10
PI = np.pi
PI_2 = np.pi * 2
PI_1_2 = np.pi * 0.5
PI_1_4 = np.pi * 0.25
RAD2DEG = 180 / np.pi
DEG2RAD = np.pi / 180


class Triangle:

    def __init__(self):
        self.vertices = []
        self.barycenter = []
        self.in_circle = []
        self.out_edges = []
        self.in_edges = []

    def set_points(self, p1, p2, p3):
        self.vertices = [p1, p2, p3]
        self.barycenter = [(p1[0] + p2[0] + p3[0]) / 3, (p1[1] + p2[1] + p3[1]) / 3]
        self.in_circle = [0, 0]
        self.out_edges = []
        self.in_edges = []

    def det_common_edge(self, other_t):
        """
        与另一个三角形对比，找共同边

        Args:
            other_t: 另一个三角形

        Returns:

        """
        for i in range(-1, len(self.vertices) - 1):
            e = [self.vertices[i], self.vertices[i + 1]]
            for j in range(-1, len(other_t.vertices) - 1):
                o_e = [other_t.vertices[j], other_t.vertices[j + 1]]
                if chk_edge_same(e, o_e):
                    return e
        return None

    def find_vertex_idx(self, v):
        for i in range(len(self.vertices)):
            if chk_p_same(self.vertices[i], v):
                return i

    def clone(self):
        p1, p2, p3 = self.vertices
        cp1, cp2, cp3 = p1[:], p2[:], p3[:]
        c_tri = Triangle()
        c_tri.set_points(cp1, cp2, cp3)
        c_tri.vertices = pickle.loads(pickle.dumps(self.vertices))
        c_tri.barycenter = pickle.loads(pickle.dumps(self.barycenter))
        c_tri.in_circle = pickle.loads(pickle.dumps(self.in_circle))
        c_tri.out_edges = pickle.loads(pickle.dumps(self.out_edges))
        c_tri.in_edges = pickle.loads(pickle.dumps(self.in_edges))
        return c_tri


def rot_vecs(v, ang):
    """
    绕(0,0)点旋转指定向量

    Args:
        v: 指定向量
        ang: 旋转角(弧度制)，+顺时针，-逆时针

    Returns: 旋转后向量

    """

    sin_t, cos_t = np.sin(ang), np.cos(ang)
    rot_mat = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
    return np.dot(v, rot_mat)


def l2_norm(v):
    """
    l2 范数

    Args:
        v: 向量

    Returns:
        长度
    """
    return (v[0] ** 2 + v[1] ** 2) ** 0.5


def l2_norm_square(v):
    return v[0] ** 2 + v[1] ** 2


def cross_2d(v1, v2):
    return v1[0] * v2[1] - v1[1] * v2[0]


def dot_2d(v1, v2):
    return v1[0] * v2[0] + v1[1] * v2[1]


def norm_vec(v):
    """
    归一化向量
    Args:
        v: 向量

    Returns:
        归一化向量
    """
    v_o = np.array(v)
    v_l = (v_o[0] ** 2 + v_o[1] ** 2) ** 0.5
    if v_l == 0:
        return np.array([0, 0])
    return np.divide(v_o, v_l)


def calc_tri_area(tri):
    """
    计算三角形面积

    Args:
        tri: 三角形顶点

    Returns:
        面积
    """
    p1x, p1y = tri[0]
    p2x, p2y = tri[1]
    p3x, p3y = tri[2]
    return (p1x * (p2y - p3y) + p2x * (p3y - p1y) + p3x * (p1y - p2y)) * 0.5


def calc_angle_bet_vec(v_base, v_target):
    """
    返回两不为0向量间夹角，从v_base转过所要计算的角度到v_target，以弧度制

    Args:
        v_base: 旋转向量
        v_target: 目标向量

    Returns:
        旋转角 0 - PI 顺时针转角 -PI - 0 逆时针转角
    """
    if (v_base[0] == 0 and v_base[1] == 0) or (v_target[0] == 0 and v_target[1] == 0):
        return 0
    vbtan = np.arctan2(v_base[1], v_base[0])
    vttan = np.arctan2(v_target[1], v_target[0])
    ang_base = vbtan if vbtan > 0 else vbtan + PI_2
    ang_tar = vttan if vttan > 0 else vttan + PI_2
    turn_ang = ang_base - ang_tar
    if turn_ang > PI:
        turn_ang -= PI_2
    elif turn_ang < -PI:
        turn_ang += PI_2
    return turn_ang


def find_ray_edge_intersect(m, p1, p2):
    """
    以m为基，向右x方向发射射线，计算射线与线段p1p2的交点

    Args:
        m:
        p1:
        p2:

    Returns:

    """
    if p1[1] > m[1] and p2[1] > m[1]:  # 线段在射线上面
        i_p = None
        t = None
    elif p1[1] < m[1] and p2[1] < m[1]:  # 线段在射线下面
        i_p = None
        t = None
    elif p1[1] == m[1] and p2[1] != m[1]:  # 线段一头在射线上，一头不在
        i_p = p1
        t = p1[0] - m[0]
    elif p2[1] == m[1] and p1[1] != m[1]:  # 线段一头在射线上，一头不在
        i_p = p2
        t = p2[0] - m[0]
    elif p1[1] == m[1] and p2[1] == m[1]:  # 线段完全在射线上
        d1 = p1[0] - m[0]
        d2 = p2[0] - m[0]
        if abs(d1) > abs(d2):  # 选距离近的点作为交点
            i_p = p2
            t = d2
        else:
            i_p = p1
            t = d1
    else:
        v = np.array(p2) - np.array(p1)
        i_p = (np.array(p1) + np.multiply(v, abs(m[1] - p1[1]) / abs(p2[1] - p1[1]))).tolist()
        t = i_p[0] - m[0]
    return i_p, t


def calc_point_mindis2poly(pos, poly: np.ndarray):
    row, col = poly.shape
    min_dis = float('inf')
    mp1, mp2 = None, None
    for i in range(-1, row - 1):
        p1, p2 = poly[i], poly[i + 1]
        dis, r, t = calc_point_pro2line(pos, p1, p2)
        if dis < min_dis:
            min_dis = dis
            mp1, mp2 = p1, p2
    return min_dis, [mp1, mp2]


def calc_point_pro2line(p, l_s, l_e):
    """
    计算点在指定线段上的投影

    Args:
        p: 点
        l_s: 线段起点
        l_e: 线段终点

    Returns:
        投影距离，投影点，投影点在线段上的t值
    """

    point_loc = np.array(p)
    line_s_loc = np.array(l_s)
    line_e_loc = np.array(l_e)
    line_vec = line_e_loc - line_s_loc
    r_dir_normal = norm_vec(line_vec)
    t = dot_2d(point_loc - line_s_loc, r_dir_normal)
    result = np.multiply(t, r_dir_normal) + l_s
    v = result - point_loc
    distance = (v[0] ** 2 + v[1] ** 2) ** 0.5
    if line_vec[0] != 0:
        t_t = t * r_dir_normal[0] / line_vec[0]
    elif line_vec[1] != 0:
        t_t = t * r_dir_normal[1] / line_vec[1]
    else:
        t_t = None
        distance = None
        result = None
    return distance, result, t_t


def calc_nonparallel_lines_intersect(l1_s, l1_e, l2_s, l2_e):
    """
    非共线线段求交点，若共线则无交点,该方法用于计算两凸多边形交集

    Args:
        l1_s: 线段1起点
        l1_e: 线段1终点
        l2_s: 线段2起点
        l2_e: 线段2终点

    Returns:
        t (# 1 - 有一个交点, 0- 无交点), p (交点), info
    """
    l1_s = np.array(l1_s)
    l1_e = np.array(l1_e)
    l2_s = np.array(l2_s)
    l2_e = np.array(l2_e)
    l1_v = l1_e - l1_s
    l2_v = l2_e - l2_s

    if cross_2d(l1_v, l2_v) != 0:  # 两线段不共线
        a, b = l1_v
        c, d = l2_v
        e, f = l2_s - l1_s
        m = a * d - c * b
        s = (e * d - c * f) / m
        t = (e * b - a * f) / m
        if 0 <= s <= 1 and 0 <= t <= 1:  # 有交点
            return 1, (np.multiply(l1_v, s) + l1_s).tolist(), (s, t)
    return 0, None, None


def calc_cir_rect(poly):
    """
    计算给定标准旋转状态下多边形最小外接矩形

    Args:
        poly: 多边形顶点列表，ndarray

    Returns:
        最小外接矩形，宽，高
    """
    [x_min, y_min] = np.min(poly, axis=0)
    [x_max, y_max] = np.max(poly, axis=0)
    min_rect = np.array([[x_min, y_min],
                         [x_max, y_min],
                         [x_max, y_max],
                         [x_min, y_max]])
    return min_rect, x_max - x_min, y_max - y_min


def calc_poly_min_cir_rect(poly):
    """
    计算给定多边形的最小外接矩形

    Args:
        p: 多边形定点序列，二维数组形式

    Returns:
        最小外接矩形，面积，多边形对应旋转角度
    """

    N, d = poly.shape
    if N < 3 or d != 2:
        raise ValueError
    rect_min, w_min, h_min = calc_cir_rect(poly)
    rad_min = 0.
    area_min = w_min * h_min
    rad = []
    for i in range(N):
        vector = poly[i - 1] - poly[i]
        rad.append(np.arctan(vector[1] / (vector[0] + EPS)))
    for r in rad:
        new_poly = rot_vecs(poly, r)
        rect, w, h = calc_cir_rect(new_poly)
        area = w * h
        if area < area_min:
            rect_min, area_min, w_min, h_min, rad_min = rect, area, w, h, -r
    min_rect_r = rot_vecs(rect_min, rad_min)
    return min_rect_r, w_min, h_min, rad_min


def calc_poly_triangulation(poly_bounds: list):
    total_bound, total_edges = calc_poly_out_bound(poly_bounds)
    tris = calc_out_bound_triangulation(total_bound, total_edges)
    return tris


def calc_poly_out_bound(poly_bounds: list):
    """
    将内部轮廓依次融合到最外层轮廓，使得最终轮廓只有一个
    Args:
        poly_bounds:

    Returns:

    """
    target_bounds = pickle.loads(pickle.dumps(poly_bounds))
    if target_bounds is None:
        return None, None
    elif len(target_bounds) == 1:  # 仅包含外轮廓
        total_bound = target_bounds[0]
        ver_num = len(total_bound)
        total_bound, _ = calc_adjust_poly_order(total_bound)
        total_edges = []
        for i in range(-1, ver_num - 1):
            total_edges.append([total_bound[i], total_bound[i + 1], 1])  # 创建轮廓边，最后一位表示是否外轮廓，用1,0表示
    else:
        bounds_num = len(target_bounds)
        out_bound = pickle.loads(pickle.dumps(target_bounds[0]))
        out_bound, _ = calc_adjust_poly_order(out_bound)  # 外轮廓调为逆时针
        print(out_bound)
        added_edges = []
        inner_bounds = pickle.loads(pickle.dumps(target_bounds[1:bounds_num]))
        while len(inner_bounds) > 0:
            selected_idx = 0
            max_x = 0
            for i in range(len(inner_bounds)):
                temp_bound = inner_bounds[i]
                max_v_x = 0
                for iv in temp_bound:
                    if iv[0] > max_v_x:
                        max_v_x = iv[0]
                if max_v_x > max_x:
                    selected_idx = i
                    max_x = max_v_x
            inner_bound = inner_bounds[selected_idx]
            inner_bound, _ = calc_adjust_poly_order(inner_bound, order=0)  # 内轮廓调为顺时针
            print(inner_bound)
            in_num = len(inner_bound)
            in_idx, out_idx = find_visible_vertex(inner_bound, out_bound)
            out_num = len(out_bound)
            out1 = out_bound[0:out_idx + 1]
            out2 = out_bound[out_idx:out_num]
            in1 = inner_bound[in_idx:in_num]
            in2 = inner_bound[0:in_idx + 1]
            out_bound = out1 + in1 + in2 + out2
            added_edges.append([out_bound[out_idx], inner_bound[in_idx]])
            added_edges.append([inner_bound[in_idx], out_bound[out_idx]])

            inner_bounds.pop(selected_idx)

        total_bound = out_bound
        ver_num = len(total_bound)
        total_edges = []
        for i in range(-1, ver_num - 1):
            is_out_e = 1
            for added_e in added_edges:
                if chk_dir_edge_same([total_bound[i], total_bound[i + 1]], added_e):
                    is_out_e = 0
                    break
            total_edges.append([total_bound[i], total_bound[i + 1], is_out_e])  # 创建轮廓边，最后一位表示是否外轮廓，用1,0表示

    return total_bound, total_edges


def calc_con_polys_intersect(poly1: list, poly1_in_cir, poly2: list, poly2_in_cir):
    """
    计算两个凸多边形交集，获得最终凸交集轮廓（按照逆时针方向排序）

    Args:
        poly1:
        poly1_in_cir:
        poly2:
        poly2_in_cir:

    Returns:
        交集区域轮廓（凸多边形）: list
    """
    if len(poly1) < 3 or len(poly2) < 3:
        inter_poly = None
    else:
        poly1_num, poly2_num = len(poly1), len(poly2)
        total_set = []
        for p in poly2:
            if chk_p_in_conv(np.array(p), poly1, poly1_in_cir):
                total_set.append(p)
        for i in range(-1, poly1_num - 1):  # 先检测poly1的所有顶点在不在poly2中,同时把poly1与poly2边的交点得到
            po1v1, po1v2 = poly1[i], poly1[i + 1]
            if chk_p_in_conv(np.array(po1v2), poly2, poly2_in_cir):
                can_add = True
                for ep in total_set:
                    if abs(ep[0] - po1v2[0]) < EPS and abs(ep[1] - po1v2[1]) < EPS:  # 判断两点是否相同，去重复
                        can_add = False
                        break
                if can_add:
                    total_set.append(po1v2)
            for j in range(-1, poly2_num - 1):
                po2v1, po2v2 = poly2[j], poly2[j + 1]
                # 这里特别注意，由于交点坐标精度问题，可能出现极小的误差，这里需要进一步处理一下，忽略误差
                i_t, i_p, _ = calc_nonparallel_lines_intersect(po1v1, po1v2, po2v1, po2v2)
                # 仅考察poly1和poly2非共线边的交叉点,原因是若两边共线，无非就是重合或完全不相交
                # 若重合，则重合点在之前判断多边形顶点是否在另一多边形内时就会找到
                if i_t:
                    can_add = True
                    for ep in total_set:
                        if abs(ep[0] - i_p[0]) < EPS and abs(ep[1] - i_p[1]) < EPS:  # 判断两点是否相同，去重复
                            can_add = False
                            break
                    if can_add:
                        total_set.append(i_p)

        # if len(total_set) >= 3:
        #     centroid = np.array(calc_convex_centroid(total_set[0:3]))
        #     pi_2 = 2 * math.pi
        #     v_angs = []
        #     for p in v_set:
        #         vec = np.array(p) - centroid
        #         ang_p = math.atan2(vec[1], vec[0])
        #         ang_p = ang_p if ang_p >= 0 else ang_p + pi_2
        #         v_angs.append(ang_p)
        #     start_i = 0
        #     while len(i_set) > 0:
        #         i_p = i_set.popleft()
        #         i_vec = np.array(i_p) - centroid
        #         i_a = math.atan2(i_vec[1], i_vec[0])
        #         i_a = i_a if i_a >= 0 else i_a + pi_2
        #         v_num = len(v_set)
        #         for i in range(start_i, v_num):
        #             v_a = v_angs[i]
        #             if i_a < v_a:
        #                 start_i = i
        #                 break
        #         v_set = v_set[0:start_i] + [i_p] + v_set[start_i:v_num]
        #         v_angs = v_angs[0:start_i] + [i_a] + v_angs[start_i:v_num]
        #     inter_poly = v_set
        # else:
        #     inter_poly = None
        inter_poly = cmp_convex_vertex_order(total_set) if len(total_set) >= 3 else None

    return inter_poly


def calc_con_polys_intersect_simple(poly1: list, poly2: list):
    """
    计算两个凸多边形交集，获得最终凸交集轮廓（按照逆时针方向排序）

    Args:
        poly1:
        poly1_in_cir:
        poly2:
        poly2_in_cir:

    Returns:
        交集区域轮廓（凸多边形）: list
    """
    if len(poly1) < 3 or len(poly2) < 3:
        inter_poly = None
    else:
        poly1_num, poly2_num = len(poly1), len(poly2)
        total_set = []
        for p in poly2:
            if chk_p_in_conv_simple(p, np.array(poly1)):
                total_set.append(p)
        for i in range(-1, poly1_num - 1):  # 先检测poly1的所有顶点在不在poly2中,同时把poly1与poly2边的交点得到
            po1v1, po1v2 = poly1[i], poly1[i + 1]
            if chk_p_in_conv_simple(po1v2, np.array(poly2)):
                can_add = True
                for ep in total_set:
                    if abs(ep[0] - po1v2[0]) < EPS and abs(ep[1] - po1v2[1]) < EPS:  # 判断两点是否相同，去重复
                        can_add = False
                        break
                if can_add:
                    total_set.append(po1v2)
            for j in range(-1, poly2_num - 1):
                po2v1, po2v2 = poly2[j], poly2[j + 1]
                # 这里特别注意，由于交点坐标精度问题，可能出现极小的误差，这里需要进一步处理一下，忽略误差
                i_t, i_p, _ = calc_nonparallel_lines_intersect(po1v1, po1v2, po2v1, po2v2)
                # 仅考察poly1和poly2非共线边的交叉点,原因是若两边共线，无非就是重合或完全不相交
                # 若重合，则重合点在之前判断多边形顶点是否在另一多边形内时就会找到
                if i_t > 0:
                    can_add = True
                    for ep in total_set:
                        if abs(ep[0] - i_p[0]) < EPS and abs(ep[1] - i_p[1]) < EPS:  # 判断两点是否相同，去重复
                            can_add = False
                            break
                    if can_add:
                        total_set.append(i_p)

        inter_poly = cmp_convex_vertex_order(total_set) if len(total_set) >= 3 else None

    return inter_poly


def calc_out_bound_triangulation(poly_bounds: list, poly_edges: list):
    """
    calculate the triangulation of a polygon, for the outer contour of polygon, the order of vertex must be
    anti-clockwise, and the inner contours are all in clockwise order.

    Args:
        poly_bounds: a list contains all bounds, [0] must be outer bound.

        poly_edges:

    Returns:
        list of triangulation
    """

    total_bound = pickle.loads(pickle.dumps(poly_bounds))
    total_edges = pickle.loads(pickle.dumps(poly_edges))
    total_num = len(total_bound)
    if total_num < 3:
        return None
    else:
        tris = []
        while len(total_bound) > 3:
            total_bound, total_edges, tri = ear_clip_poly_opti(total_bound, total_edges)  # or use 'ear_clip_poly'
            tris.append(tri)
        tri = Triangle()
        tri.set_points(total_bound[0], total_bound[1], total_bound[2])
        tri.vertices = cmp_convex_vertex_order(tri.vertices)
        tri.in_circle = calc_poly_max_in_circle(tri.vertices)
        e1 = remove_edge_from_edgeset(total_edges, [total_bound[0], total_bound[1]])
        e2 = remove_edge_from_edgeset(total_edges, [total_bound[1], total_bound[2]])
        e3 = remove_edge_from_edgeset(total_edges, [total_bound[0], total_bound[2]])
        if e1[2]:
            tri.out_edges.append([tri.find_vertex_idx(e1[0]), tri.find_vertex_idx(e1[1])])
        else:
            tri.in_edges.append([tri.find_vertex_idx(e1[0]), tri.find_vertex_idx(e1[1])])
        if e2[2]:
            tri.out_edges.append([tri.find_vertex_idx(e2[0]), tri.find_vertex_idx(e2[1])])
        else:
            tri.in_edges.append([tri.find_vertex_idx(e2[0]), tri.find_vertex_idx(e2[1])])
        if e3[2]:
            tri.out_edges.append([tri.find_vertex_idx(e3[0]), tri.find_vertex_idx(e3[1])])
        else:
            tri.in_edges.append([tri.find_vertex_idx(e3[0]), tri.find_vertex_idx(e3[1])])
        tris.append(tri)
        return tris


def calc_adjust_poly_order(poly_bound: list, order=1):
    """
    将多边形轮廓点按照指定方向重新调整，改变为顺时针/逆时针，默认逆时针. 1. 先利用鞋带公式求原始轮廓的方向. 2. 调整为目标方向

    Args:
        poly_bound: 轮廓点

        order: 1-逆时针，0-顺时针

    Returns:
        调整后的轮廓点
    """
    ori_order = chk_poly_order(poly_bound)
    if (order and not ori_order) or (not order and ori_order):
        poly_bound.reverse()
    return poly_bound, ori_order


def find_visible_vertex(inner_bound, outer_bound):
    """
    返回指定内部轮廓与外部轮廓间的一对相互可见顶点，返回值是相应顶点在各自轮廓中的index

    Args:
        inner_bound:
        outer_bound:

    Returns:
        内轮廓可见点idx，外轮廓可见点idx
    """
    M = None
    m_idx = 0
    mx_max = 0
    in_num = len(inner_bound)
    for j in range(0, in_num):
        iv = inner_bound[j]
        if iv[0] >= mx_max:
            M = iv
            m_idx = j
            mx_max = iv[0]
    out_num = len(outer_bound)
    intersect_t = float('inf')
    I, cor_b1, cor_b2 = None, None, None
    cor_b1_idx, cor_b2_idx = 0, 0
    for i in range(-1, out_num - 1):
        ov1 = outer_bound[i]
        ov2 = outer_bound[i + 1]
        i_p, i_t = find_ray_edge_intersect(M, ov1, ov2)
        if i_p is not None:  # 有交点
            if 0 <= i_t <= intersect_t:
                I, cor_b1, cor_b2 = i_p, ov1, ov2
                cor_b1_idx, cor_b2_idx = i, i + 1
                intersect_t = i_t
    if chk_p_same(I, cor_b1):  # 如果交点就是外轮廓上的一个点
        return m_idx, cor_b1_idx
    elif chk_p_same(I, cor_b2):  # 如果交点就是外轮廓上的一个点
        return m_idx, cor_b2_idx
    else:  # 焦点在外轮廓的一条边上，需要进一步处理
        if cor_b1[0] > cor_b2[0]:
            P = cor_b1
            p_idx = cor_b1_idx
        else:
            P = cor_b2
            p_idx = cor_b2_idx
        p_in_MIP = []
        tri_c = np.array([(M[0] + I[0] + P[0]) / 3, (M[1] + I[1] + P[1]) / 3])  # 当前三个点构成三角形的重心
        nM, nI, nP = np.array(M), np.array(I), np.array(P)
        for r_i in range(0, out_num - 2):
            idx = (cor_b2_idx + r_i + 1) % out_num
            other_v = np.array(outer_bound[idx])  # 除刚才三个点外的其他点
            if chk_ps_on_line_side(tri_c, other_v, nM, nI):
                if chk_ps_on_line_side(tri_c, other_v, nI, nP):
                    if chk_ps_on_line_side(tri_c, other_v, nP, nM):
                        p_in_MIP.append(idx)
        if len(p_in_MIP) == 0:  # 如果没有别的外轮廓点在三角形MIP中，则MP构成一对相互可见点
            return m_idx, p_idx
        else:  # 存在，则从中找出与x轴夹角最小的点与M构成一对相互可见点
            min_a = float('inf')
            for p_i in p_in_MIP:
                potential_p = outer_bound[p_i]
                v_mp = np.array(potential_p) - np.array(M)
                v_len = l2_norm(v_mp)
                if v_len < min_a:
                    p_idx = p_i
                    min_a = v_len
            return m_idx, p_idx


def ear_clip_poly_opti(bound, edges):
    """
        从给定轮廓中用耳切法切掉一个耳朵

        Args:
            bound:
            edges:

        Returns:

        """
    cut_ear = None
    bound_num = len(bound)
    ears = []
    for i in range(0, bound_num):
        is_ear = True
        v_f = bound[i - 1]  # vn-1
        v = bound[i]  # vn
        v_n = bound[(i + 1) % bound_num]  # vn+1
        nv_f = np.array(v_f)
        nv = np.array(v)
        nv_n = np.array(v_n)
        vec1 = np.array([v[0] - v_f[0], v[1] - v_f[1]])
        vec2 = np.array([v_n[0] - v[0], v_n[1] - v[1]])
        if calc_angle_bet_vec(vec2, vec1) <= 0:  # 当前拐角不是凸的
            is_ear = False
        else:  # 当前拐角是凸的，可能是耳朵
            tri_c = np.array([(v_f[0] + v[0] + v_n[0]) / 3, (v_f[1] + v[1] + v_n[1]) / 3])  # 当前三个点构成三角形的重心
            for j in range(0, bound_num - 3):
                other_v = np.array(bound[(i + j + 2) % bound_num])  # 除刚才三个点外的其他点
                in_tri = False
                if chk_ps_on_line_side(tri_c, other_v, nv_f, nv):
                    if chk_ps_on_line_side(tri_c, other_v, nv, nv_n):
                        if chk_ps_on_line_side(tri_c, other_v, nv_n, nv_f):
                            in_tri = True
                if in_tri:  # 存在轮廓点在当前您的三角中，代表这个不是耳朵
                    is_ear = False
                    break
        if is_ear:
            ears.append(i)
    target_idx = 0
    tar_vf, tar_v, tar_vn = None, None, None
    max_area = 0  # 选面积最大的耳朵
    for idx in ears:
        v_f = bound[idx - 1]  # vn-1
        v = bound[idx]  # vn
        v_n = bound[(idx + 1) % bound_num]  # vn+1
        # vec1 = np.array([v[0] - v_f[0], v[1] - v_f[1]])
        # vec2 = np.array([v_n[0] - v[0], v_n[1] - v[1]])
        # ang = calc_angle_bet_vec(vec2, vec1)
        # if ang < min_angle:
        #     target_idx = idx
        #     tar_vf, tar_v, tar_vn = v_f, v, v_n
        #     min_angle = ang
        area = calc_tri_area([v_f, v, v_n])
        if area > max_area:
            target_idx = idx
            tar_vf, tar_v, tar_vn = v_f, v, v_n
            max_area = area
    bound.pop(target_idx)
    e1 = remove_edge_from_edgeset(edges, [tar_vf, tar_v], is_dir=1)
    e2 = remove_edge_from_edgeset(edges, [tar_v, tar_vn], is_dir=1)
    e3 = [tar_vf, tar_vn, 0]  # 创建地边一定是内部边
    edges.append(e3)
    cut_ear = Triangle()
    cut_ear.set_points(tar_vf, tar_v, tar_vn)
    cut_ear.vertices = cmp_convex_vertex_order(cut_ear.vertices)
    cut_ear.in_circle = calc_poly_max_in_circle(cut_ear.vertices)
    if e1[2]:
        cut_ear.out_edges.append([cut_ear.find_vertex_idx(e1[0]), cut_ear.find_vertex_idx(e1[1])])
    else:
        cut_ear.in_edges.append([cut_ear.find_vertex_idx(e1[0]), cut_ear.find_vertex_idx(e1[1])])
    if e2[2]:
        cut_ear.out_edges.append([cut_ear.find_vertex_idx(e2[0]), cut_ear.find_vertex_idx(e2[1])])
    else:
        cut_ear.in_edges.append([cut_ear.find_vertex_idx(e2[0]), cut_ear.find_vertex_idx(e2[1])])
    cut_ear.in_edges.append([cut_ear.find_vertex_idx(e3[0]), cut_ear.find_vertex_idx(e3[1])])

    return bound, edges, cut_ear


def calc_convex_centroid(convex_set):
    """
    求解凸多边形质心

    Args:
        convex_set:

    Returns:

    """
    row, col = convex_set.shape
    if row < 2:
        return None
    else:
        return np.divide(np.sum(convex_set, axis=0), row)


def calc_poly_max_in_circle(poly: list):
    tar_poly = pickle.loads(pickle.dumps(poly))
    x_sub, y_sub = 20, 20
    stop_thre = 0.1
    x_min, x_max = float('inf'), 0
    y_min, y_max = float('inf'), 0
    for p in tar_poly:
        x, y = p
        if x < x_min:
            x_min = x
        if x > x_max:
            x_max = x
        if y < y_min:
            y_min = y
        if y > y_max:
            y_max = y
    bound = [x_min, x_max, y_min, y_max]
    interval = (2 ** 0.5) * 2

    while True:
        c_center, radius = calc_poly_in_circle(tar_poly, bound, x_sub, y_sub)
        cx, cy = c_center
        fit_tmp = (bound[1] - bound[0]) / interval
        bound[1] = cx + fit_tmp
        bound[0] = cx - fit_tmp
        fit_tmp = (bound[3] - bound[2]) / interval
        bound[3] = cy + fit_tmp
        bound[2] = cy - fit_tmp
        if (bound[1] - bound[0]) < stop_thre or (bound[3] - bound[2]) < stop_thre:
            break

    return c_center, radius


def calc_poly_in_circle(poly: list, out_bound, x_sub, y_sub):
    poly = np.array(poly)
    c_center = [0, 0]
    x_inc = (out_bound[1] - out_bound[0]) / x_sub
    y_inc = (out_bound[3] - out_bound[2]) / y_sub
    max_dis = 0
    for i in range(x_sub):
        x_temp = out_bound[0] + i * x_inc
        for j in range(y_sub):
            y_temp = out_bound[2] + j * y_inc
            if chk_p_in_conv_simple([x_temp, y_temp], poly):
                dis_temp, _ = calc_point_mindis2poly([x_temp, y_temp], poly)
                if dis_temp > max_dis:
                    max_dis = dis_temp
                    c_center = [x_temp, y_temp]
    return c_center, max_dis


def cmp_convex_vertex_order(points_set: list):
    """
    对多边形顶点进行排序，排序按照逆时针排序

    Args:
        points_set:

    Returns:

    """
    centroid = calc_convex_centroid(np.array(points_set[0:3]))
    point_num = len(points_set)
    point_angs = []
    reordered_poly = []
    reordered_angs = []
    for p in points_set:
        vec = np.array(p) - centroid
        ang_p = np.arctan2(vec[1], vec[0])
        ang_p = ang_p if ang_p >= 0 else ang_p + PI_2
        point_angs.append(ang_p)
    #     if len(reordered_poly) == 0:
    #         reordered_poly.append(p)
    #         reordered_angs.append(ang_p)
    #     else:
    #         r_num = len(reordered_poly)
    #         insert_i = 0
    #         for i in range(r_num):
    #             ang_rp = reordered_angs[i]
    #             insert_i = i
    #             if ang_rp > ang_p:
    #                 break
    #         if insert_i is not None:
    #             reordered_poly = reordered_poly[0:insert_i] + [p] + reordered_poly[insert_i:r_num + 1]
    #             reordered_angs = reordered_angs[0:insert_i] + [ang_p] + reordered_angs[insert_i:r_num + 1]
    # return reordered_poly

    for i in range(0, point_num - 1):
        for j in range(0, point_num - i - 1):
            ang1 = point_angs[j]
            ang2 = point_angs[j + 1]
            if ang1 > ang2:  # 点p1以多边形质心centroid逆时针绕过的角度大于点p2，这代表p1应该排在p2的后面
                temp = np.array(points_set[j]).tolist()
                points_set[j] = points_set[j + 1]
                points_set[j + 1] = temp
                temp_p = point_angs[j]
                point_angs[j] = point_angs[j + 1]
                point_angs[j + 1] = temp_p
    return points_set


def chk_p_in_conv(pos, poly: list, poly_in_circle):
    """
    计算给定点是否在指定的凸多边形内，在边界上也算在多边形内，需要多边形必须是逆时针排序

    Args:

        pos:
        poly:
        poly_in_circle:

    Returns:
    """

    c, r = poly_in_circle
    c = np.array(c)
    v = pos - c
    if v[0] ** 2 + v[1] ** 2 <= r ** 2:
        return True
    else:
        n_poly = np.array(poly)
        n_poly = n_poly - pos
        poly_num = len(poly)
        for i in range(-1, poly_num - 1):
            pv1, pv2 = n_poly[i], n_poly[i + 1]
            cross = cross_2d(pv1, pv2)
            if cross < 0:  # 点在多边形边右侧，代表在多边形之外
                return False
            elif cross == 0 and dot_2d(pv1, pv2) <= 0:  # 点p在直线v1v2上，并且在线段v1v2之间，则直接判定在多边形内
                return True
        return True


def chk_p_in_conv_simple(pos, poly: np.ndarray):
    """
    计算给定点是否在指定的凸多边形内，在边界上也算在多边形内，需要多边形必须是逆时针排序

    Args:
        pos:
        poly:

    Returns:

    """

    row, col = poly.shape
    p_poly = poly - pos
    for i in range(-1, row - 1):
        pv1, pv2 = p_poly[i], p_poly[i + 1]
        cross = cross_2d(pv1, pv2)
        if cross < 0:  # 点在多边形边右侧，代表在多边形之外
            return False
        elif cross == 0 and dot_2d(pv1, pv2) <= 0:  # 点p在直线v1v2上，并且在线段v1v2之间，则直接判定在多边形内
            return True
    return True


def chk_poly_order(poly_bound: list):
    """
    检查多边形顺逆时针方向

    Args:
        poly_bound:

    Returns:
        1-逆时针 0-顺时针
    """
    s = 0
    poly_num = len(poly_bound)
    for i in range(-1, poly_num - 1):
        s += (poly_bound[i + 1][1] + poly_bound[i][1]) * (poly_bound[i][0] - poly_bound[i + 1][0])
    ori_order = 1 if s > 0 else 0
    return ori_order


def chk_ps_on_line_side(p1, p2, l_s, l_e):
    """
    检测两点是否位于线段同侧

    Args:
        p1: 点1
        p2: 点2
        l_s: 线段起点
        l_e: 线段终点

    Returns:
        同侧-1 不同侧-0
    """
    return cross_2d(p1 - l_s, p1 - l_e) * cross_2d(p2 - l_s, p2 - l_e) > 0


def remove_edge_from_edgeset(edge_set, tar, is_dir=0):
    """
    从指定边集中检测并删除目标边，若有则返回删除边，若无则返回空. 默认针对无向边

    Args:
        edge_set:
        tar:
        is_dir:

    Returns:

    """
    if tar is None:
        return None
    elif not is_dir:
        for e in edge_set:
            if chk_edge_same(e, tar):
                edge_set.remove(e)
                return e
        return None
    else:
        for e in edge_set:
            if chk_dir_edge_same(e, tar):
                edge_set.remove(e)
                return e
        return None


def chk_dir_edge_same(e1, e2):
    """
    判断无向边是否相同

    Args:
        e1:
        e2:

    Returns:

    """
    if chk_p_same(e1[0], e2[0]) and chk_p_same(e1[1], e2[1]):
        return True
    else:
        return False


def chk_edge_same(e1, e2):
    """
    判断无向边是否相同

    Args:
        e1:
        e2:

    Returns:

    """
    if chk_p_same(e1[0], e2[0]) and chk_p_same(e1[1], e2[1]):
        return True
    elif chk_p_same(e1[0], e2[1]) and chk_p_same(e1[1], e2[0]):
        return True
    else:
        return False


def chk_p_same(p1, p2):
    return abs(p1[0] - p2[0]) < EPS and abs(p1[1] - p2[1]) < EPS
