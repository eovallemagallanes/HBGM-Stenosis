import random
import math
from PIL import Image, ImageDraw, ImageFilter
from scipy.stats import norm
import matplotlib.pyplot as plt
import numpy as np
import cv2

#set seed
#random.seed(11)


def make_bezier(xys, ts=100):
    # xys should be a sequence of 2-tuples (Bezier control points)
    n = len(xys)
    combinations = pascal_row(n - 1)
    ts = [t / ts for t in range(ts+1)]

    # This uses the generalized formula for bezier curves
    # http://en.wikipedia.org/wiki/B%C3%A9zier_curve#Generalization
    result = []
    for t in ts:
        tpowers = (t ** i for i in range(n))
        upowers = reversed([(1 - t) ** i for i in range(n)])
        coefs = [c * a * b for c, a, b in zip(combinations, tpowers, upowers)]
        result.append(
            tuple(sum([coef * p for coef, p in zip(coefs, ps)]) for ps in zip(*xys)))
    return result


def pascal_row(n):
    # This returns the nth row of Pascal's Triangle
    result = [1]
    x, numerator = 1, n
    for denominator in range(1, n // 2 + 1):
        # print(numerator,denominator,x)
        x *= numerator
        x /= denominator
        result.append(x)
        numerator -= 1
    if n & 1 == 0:
        # n is even
        result.extend(reversed(result[:-1]))
    else:
        result.extend(reversed(result))
    return result


def draw_gradient(image):
    inner_color = random.randint(0, 25) + 25
    outer_color = random.randint(60, 80) + 25

    max_dist = math.sqrt((image.width * 2) ** 2 + (image.height * 2) ** 2)
    # print('max_dist: ', max_dist)

    center = [random.randint(-image.width, image.width * 2), random.randint(-image.height, image.height * 2)]
    for y in range(image.height):
        for x in range(image.width):
            # Find the distance to the center

            distance_to_center = math.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)

            # Make it on a scale from 0 to 1
            distance_to_center = float(distance_to_center) / max_dist  # (math.sqrt(2) * image.width / 2)
            # print('distance_to_center: ', distance_to_center)

            # Calculate r, g, and b values
            color = outer_color * distance_to_center + inner_color * (1 - distance_to_center)

            # Place the pixel
            image.putpixel((x, y), int(color))


def add_white_noise(image, delta):
    noise = Image.new('L', (image.width, image.height), 0)
    for x in range(0, image.width):
        for y in range(0, image.height):
            col = random.randint(-delta, delta)
            color = image.getpixel((x, y))
            image.putpixel((x, y), color + col)
            noise.putpixel((x, y), col)

    return noise


def make_mainvessel(width, height):
    points = 0
    return points


def make_bifurcation(point, height, width):
    points = 0
    return points


def draw_vessel(w, h, points, basic_width, with_stenosis, color_base, mark_stenosis=False):
    color_stenosis = (255, 0, 0, 255)
    poly = Image.new('RGBA', (w, h))
    pdraw = ImageDraw.Draw(poly)
    mu = random.randint(len(points) // 4, len(points) - len(points) // 4)
    sigma = random.randint(4, 8)
    scale = 10

    for index, point in enumerate(points):
        stenosis_factor = max(0.5, 1 - norm.pdf(index, mu, sigma) * scale) if with_stenosis else 1
        if stenosis_factor > 1.0:
            stenosis_factor = 1.0

        width = basic_width * stenosis_factor

        if stenosis_factor < 0.95 and mark_stenosis:
            color = color_stenosis
        else:
            color = color_base

        pdraw.ellipse((point[0] - width, point[1] - width, point[0] + width, point[1] + width), fill=color)

    return poly



def create_tree(image, h, w, parents_widths, childs_widths, parent_stenosis, childs_stenosis):
    image_seg = Image.new('L', (w, h), 0)
    color = (0, 0, 0, 100)
    color_seg = (255, 255, 255, 255)

    for k, (p, p_stenosis) in enumerate(zip(parents_widths, parent_stenosis)):
        # create parents vessels
        start_point = (random.randint(-w, w * 2), 0)
        inter_point = (random.randint(0, w), random.randint(0, h))
        inter_point2 = (random.randint(0, w), random.randint(0, h))
        end_point = (random.randint(-w, w * 2), h)

        p_points = make_bezier([start_point, inter_point, inter_point2, end_point])

        #create each parent's child
        for p_childs, p_stenosis_childs in zip(childs_widths[k], childs_stenosis[k]):
            # coordinates to draw a bifurcation
            # get a random point from the first set of points to start now from them the next curve
            new_pos = random.randint(0, len(p_points) - 1)
            y_point = p_points[new_pos]
            y_point = (int(y_point[0]), int(y_point[1]))

            start_point_b = y_point
            inter_point_b = (
                random.randint(y_point[0], y_point[0] + w),
                random.randint(y_point[1], y_point[1] + h))
            inter_point2_b = (
                random.randint(y_point[0], y_point[0] + w),
                random.randint(y_point[1], y_point[1] + h))
            end_point_b = (random.randint(-w, w*2), h)

            child_points = make_bezier([start_point_b, inter_point_b, inter_point2_b, end_point_b])
            color_c = (0, 0, 0, 100)
            poly_child = draw_vessel(w, h, child_points, p_childs, p_stenosis_childs, color_c)
            poly_child_seg = draw_vessel(w, h, child_points, p_childs, p_stenosis_childs, color_seg, mark_stenosis=True)
            image.paste(poly_child, mask=poly_child)
            image_seg.paste(poly_child_seg, mask=poly_child_seg)

        # DRAW PARENT
        poly = draw_vessel(w, h, p_points, p, p_stenosis, color)
        poly_seg = draw_vessel(w, h, p_points, p, p_stenosis, color_seg, mark_stenosis=True)
        image.paste(poly, mask=poly)
        image_seg.paste(poly_seg, mask=poly_seg)

    return image, image_seg


def get_image(w, h, th_vessel, th_stenosis):
    vessel_percent = 0
    stenosis_case = False
    # iterate if vessel percent if lower that a threshold
    while vessel_percent < th_vessel:
        image = Image.new('L', (w, h), 80)
        draw_gradient(image)
        # image = create_tree(image, w, h, [2, 2], [[1, 1], [1]], [True, False], [[False, False], [False]])

        n_parents = random.randint(1, 1)
        n_childs = [random.randint(0, 2) for _ in range(n_parents)]
        p_widths = [random.randint(1, 4) for _ in range(n_parents)]
        p_stenosis = [bool(random.getrandbits(1)) for _ in range(n_parents)]
        c_widths = []
        c_stenosis = []
        for k, p_w in enumerate(p_widths):
            c_widths.append([max(p_w * random.random(), 1) for _ in range(n_childs[k])])
            if c_with_stenosis:
                c_stenosis.append([bool(random.getrandbits(1)) for _ in range(n_childs[k])])
            else:
                c_stenosis.append([False for _ in range(n_childs[k])])

        image, image_seg = create_tree(image, h, w, p_widths, c_widths, p_stenosis, c_stenosis)
        add_white_noise(image, 16)
        image = image.filter(ImageFilter.GaussianBlur(random.randint(7, 8) * 0.1))
        add_white_noise(image, 1)

        # count vessel pixels
        image_seg2 = np.asarray(image_seg)
        num_pix_veins = np.count_nonzero(image_seg2)
        vessel_percent = num_pix_veins / (w * h)

        num_pix_stenosis = sum(sum(image_seg2 == 76))  # COLOR STENOSIS
        stenosis_percent = num_pix_stenosis / num_pix_veins


        if stenosis_percent > th_stenosis:
            stenosis_case = True

        else:
            image_seg2 = np.where(image_seg2 == 76, 255, image_seg2)
            stenosis_case = False

        return  np.asarray(image), image_seg2, stenosis_case


#w, h = 64, 64
w, h = 32, 32
th_vessel = 0.10
th_stenosis = 0.15
c_with_stenosis = False
PATH_XCA = '../HBGM-DATASET'
PATH_XCA_SEG = '../HBGM-DATASET-SEG'
pos_cases = 10
neg_cases = 10
generate = True
i = 0
j = 0
# iterate to generate images
while generate:
    xca, xca_seg, stenosis_case = get_image(w, h, th_vessel, th_stenosis)
    if stenosis_case and i<pos_cases:
        cv2.imwrite(PATH_XCA+'/positive/%06d.png'%i, xca)
        cv2.imwrite(PATH_XCA_SEG+'/positive/%06d.png'%i, xca_seg)
        i = i + 1
    else:
        if j<neg_cases:
            cv2.imwrite(PATH_XCA+'/negative/%06d.png'%j, xca)
            cv2.imwrite(PATH_XCA_SEG+'/negative/%06d.png'%j, xca_seg)
            j = j+1

    if i>=pos_cases and j>=neg_cases:
        break






