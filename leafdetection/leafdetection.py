#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Leafdetection, a tool to automatically detect leafs
# Copyright 2018 Tobias Jachowski
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
The module Leafdetection provides functions to automatically detect leafs.

Leafdetection provides functions to automatize the process of detecting leafs
in RGB images and review and revise the detected leafs. The results are stored
to be readily processed by the "Automated Colorimetric Assay" software, as
described in [1].

[1] Bresson, J., Bieker, S., Riester, L., Doll, J., & Zentgraf, U. (2017).
    A guideline for leaf senescence analyses: from quantification to
    physiological and molecular investigations. Journal of experimental botany.
    doi:10.1093/jxb/ery011

Examples
--------
####
# Automatically detect leafs in several JPG images, reorder them and overwrite
# old results with reordered leafs
# See also the included notebook 'leafdetection.py'
####

# Import the necessary functions for autodetection
from matplotlib import pyplot as plt
from leafdetection import create_experiment_name, autodetect_leafs
from leafdetection import reorder_leafs, overwrite_with_reordered_leafs

# Set some variables

# Adjust the threshold for leaf detection
threshold = 19

# Adjust the minimum area a detected leaf has to have to be
# accepted as one. You have to consider:
#   1. Leaf size
#   2. Image size (4000 x 6000 pixel)
#   3. Zoom factor of camera
min_area = 5000

# horizontal_dilate needs to be larger than half the difference
# of the x values of the central positions of two neighbouring
# leafs (in one row), but as small as possible.
horizontal_dilate = 400

# vertical_dilate needs to be larger than half the difference
# of the y values of the central positions of two neighbouring
# leafs (in one row), but as small as possible.
vertical_dilate = 200

# Select the directory, where the leaf images to be processed are located
leaf_images_dir = './images'

# Select the directories the images, regions and overview should be saved into
results_dir = './results'

# Autodetect leafs of all 'JPG' images in the folder `leaf_images_dir`
for dirpath, dirnames, image_filenames in os.walk(leaf_images_dir):
    for image_filename in image_filenames:
        if image_filename.endswith('.JPG') or image_filename.endswith('.jpg'):
            fullname = os.path.abspath(os.path.join(dirpath, image_filename))
            experiment_name = create_experiment_name(image_filename)
            autodetect_leafs(fullname, results_dir, experiment_name,
                             threshold=threshold, min_area=min_area,
                             vertical_dilate=vertical_dilate,
                             horizontal_dilate=horizontal_dilate,
                             verbose=False)

# Reorder autodetected leafs from original image IMG_3576.JPG
image_filename = './images/IMG_3576.JPG'
results_dir = 'results'
experiment_name = create_experiment_name(image_filename)
fig, ax, leaf_regions, regions_removed = reorder_leafs(image_filename,
                                                       results_dir,
                                                       experiment_name)

# ... reorder the leafs in the plotted figure first and then, to overwrite the
# the old results with reordered leafs, execute:
overwrite_with_reordered_leafs(fig, image_filename, results_dir,
                               experiment_name, leaf_regions)

# close the figure and release the memory
fig.clear()
plt.close(fig)
"""

__author__ = "Tobias Jachowski"
__copyright__ = "Copyright 2018 Tobias Jachowski"
__credits__ = "Gero Hermsdorf, Steve Simmert"
__license__ = "Apache-2.0"
__version__ = "1.0.0"
__maintainer__ = "Tobias Jachowski"
__email__ = "leafdetection@jachowski.de"
__status__ = "stable"

import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os
import skimage.io

from matplotlib.gridspec import GridSpec
from skimage.color import label2rgb
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops
from skimage.morphology import binary_closing, disk
from skimage.segmentation import clear_border


def range_overlap(a_min, a_max, b_min, b_max):
    """
    Check, if ranges [a_min, a_max] and [b_min, b_max] overlap.
    """
    return (a_min <= b_max) and (b_min <= a_max)


def region_overlap(r1, r2, row_dilate=0, col_dilate=0, centroid=True):
    """
    Check, if two regions overlap.
    """
    if centroid:
        row_1, col_1 = r1.centroid
        min_row_1 = row_1
        max_row_1 = row_1
        min_col_1 = col_1
        max_col_1 = col_1
        row_2, col_2 = r2.centroid
        min_row_2 = row_2
        max_row_2 = row_2
        min_col_2 = col_2
        max_col_2 = col_2
    else:
        min_row_1, min_col_1, max_row_1, max_col_1 = r1.bbox
        min_row_2, min_col_2, max_row_2, max_col_2 = r2.bbox

    min_row_1 -= row_dilate
    max_row_1 += row_dilate
    min_col_1 -= col_dilate
    max_col_1 += col_dilate
    min_row_2 -= row_dilate
    max_row_2 += row_dilate
    min_col_2 -= col_dilate
    max_col_2 += col_dilate

    return (range_overlap(min_row_1, max_row_1, min_row_2, max_row_2)
            and range_overlap(min_col_1, max_col_1, min_col_2, max_col_2))


def group_and_sort_regions(regions, vertical_dilate=0, horizontal_dilate=0,
                           centroid=True, inplace=True, verbose=False):
    """
    Group regions of an image according to their arrangement in rows
    and sort them according to their order in:
      1. rows
      2. columns
    """
    if not inplace:
        regions = regions.copy()

    # Index of groups, the regions have to be assigned to
    group = 0
    region_groups = []

    # List of regions not assigned to a group
    for r in regions:
        try:
            del r.group
            del r.index
        except AttributeError:
            pass

    # Sort regions according to column (x value) of central position
    # so that the first region of every new group to be created will be
    # the outer most left region.
    regions.sort(key=lambda region: region.centroid[1])
    regions_ungrouped = regions.copy()

    # Compare all regions with each of the other regions.
    for r1 in regions:
        if not hasattr(r1, 'group'):
            # r1 is not yet assigned to a group
            # r1 is the outer most left (first) region of a group
            # Assign r1 to a new group
            r1.group = group
            region_groups.append([r1])
            group += 1
            # Delete r1 from list of ungrouped regions
            grouped_region = regions_ungrouped.pop(0)
            # print('new group: ', grouped_region.group,
            #       grouped_region.centroid)
        for i2, r2 in enumerate(regions_ungrouped):
            # Find the right neighbout of r1
            if region_overlap(r1, r2, row_dilate=vertical_dilate,
                              col_dilate=horizontal_dilate, centroid=centroid):
                # r2 is the right neighbor of r1. Assign r2 to the
                # same group as r1
                r2.group = r1.group
                region_groups[r1.group].append(r2)
                # Delete r2 from list of ungrouped regions
                grouped_region = regions_ungrouped.pop(i2)
                # print('into group: ', grouped_region.group,
                #       grouped_region.centroid)
                break

    # Sort groups of regions according to row (y value) of central position of
    # first region in each group
    region_groups.sort(key=lambda region: region[0].centroid[0])

    # Set group and index of regions according to sorting order
    r_i = 0
    for r_g, g in enumerate(region_groups):
        for r in g:
            r.group = r_g
            r.index = r_i
            r_i += 1

    # Sort original regions array according to 1. group and 2. index
    regions.sort(key=lambda region: (region.group, region.index))

    if verbose:
        print('Detected {} leafs in {} rows.'.format(r_i, group))

    return regions


def detect_segments(image, threshold='auto', closing=True, selem=None):
    if selem is None:
        selem = disk(3)
    # Preprocess image (subtract blue from green channel) and show
    image_diff = image[:, :, 1] - image[:, :, 2]
    # Set all pixels to zero where there was an 8bit integer overflow (> 255)
    image_diff[image_diff > image[:, :, 1]] = 0

    # Detect leafs and create new figure with leafs to be labeled by the user
    # Apply threshold
    if threshold == 'auto':
        thresh = threshold_otsu(image_diff)
    else:
        thresh = threshold
    # Dilate and erode, to remove small dark spots
    if closing:
        mask = binary_closing(image_diff > thresh, selem)
    else:
        mask = image_diff > thresh
    # Remove artifacts connected to image border
    cleared_mask = clear_border(mask)

    return cleared_mask, label(cleared_mask)


def create_region_name(region):
    min_row, min_col, max_row, max_col = region.bbox
    name = '{:02d}_{:03d}_{}_{}_{}_{}'.format(region.group + 1,
                                              region.index + 1,
                                              min_row, min_col,
                                              max_row, max_col)
    return name


def create_experiment_name(image_filename):
    basename = os.path.basename(image_filename)
    name, ext = os.path.splitext(basename)
    try:
        components = name.split('@')
        line = components[0]
        plant = components[1]
        week = components[2]
        name = 'Line {} - Plant {} - WAS {}'.format(line, plant, week)
    except IndexError:
        pass

    return name


def create_image_name(experiment_name, region):
    leaf = region.index + 1
    name = '{} - Leaf {}'.format(experiment_name, leaf)
    return name


class Region(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self


def check_dir(directory, autocreate=False):
    absdir = os.path.abspath(directory)

    if autocreate:
        # Create the directory
        if not os.path.isdir(absdir):
            os.mkdir(absdir)

    return absdir


def delete_regions(directory, experiment_name, suffix='.npy'):
    filenames = [os.path.join(directory, f)
                 for f in os.listdir(directory)
                 if (f.startswith(experiment_name)
                     and f.endswith(suffix))]
    for filename in filenames:
        os.remove(filename)


def delete_region_images(directory, experiment_name, suffix='.png'):
    filenames = [os.path.join(directory, f)
                 for f in os.listdir(directory)
                 if (f.startswith(experiment_name)
                     and f.endswith(suffix))]
    for filename in filenames:
        os.remove(filename)


def delete_reorder_figure(directory, experiment_name, extension='reordered',
                          suffix='.jpg'):
    end = ''.join((extension, suffix))
    filenames = [os.path.join(directory, f)
                 for f in os.listdir(directory)
                 if (f.startswith(experiment_name)
                     and f.endswith(end))]
    for filename in filenames:
        os.remove(filename)


def save_regions(regions, directory, experiment_name, suffix='.npz'):
    filename = '{}@regions{}'.format(experiment_name, suffix)
    fullname = os.path.join(directory, filename)
    regions_dict = {}
    for region in regions:
        name = create_region_name(region)
        regions_dict[name] = region.image
    np.savez_compressed(fullname, **regions_dict)


def read_regions(directory, experiment_name, suffix='.npz'):
    regions = []
    filename = os.path.join(directory,
                            '{}@regions{}'.format(experiment_name, suffix))
    loaded = np.load(filename)
    for name, image in loaded.items():
        components = [int(component) for component in name.split('_')]
        group, number, min_row, min_col, max_row, max_col = components
        bbox = min_row, min_col, max_row, max_col

        rows, columns = np.where(image)
        coords = np.c_[rows + min_row, columns + min_col]

        region = Region(group=group - 1, index=number - 1, image=image,
                        bbox=bbox, coords=coords)
        regions.append(region)

    regions.sort(key=lambda region: region.index)
    return regions


def create_overview_figure(images, regions, figure_size=8, bbox_border=5):
    fig = plt.figure(figsize=[figure_size, figure_size])
    fig.subplots_adjust(wspace=0, hspace=0, left=0.0, right=1.0, top=1.0,
                        bottom=0.0)
    height = 1000
    gs = GridSpec(height, 2)
    axes = []
    height_part = 333
    axes.append(fig.add_subplot(gs[:height_part + 1, 0]))
    axes.append(fig.add_subplot(gs[:height_part + 1, 1]))
    axes.append(fig.add_subplot(gs[height_part:, :]))
    for ax in axes:
        ax.axis('off')

    for ax, image in zip(axes, images):
        ax.imshow(image)

    draw_rectangles(axes[2], regions, bbox_border=bbox_border)

    return fig, axes


def create_reorder_figure(image, regions, figure_size=8, bbox_border=5):
    aspect = image.shape[0] / image.shape[1]
    fig, ax = plt.subplots(figsize=[figure_size, figure_size*aspect])
    fig.subplots_adjust(wspace=0, hspace=0, left=0.0, right=1.0, top=1.0,
                        bottom=0.0)
    ax.axis('off')

    ax.imshow(image)

    draw_rectangles(ax, regions, bbox_border=bbox_border)

    # List of to be removed leafs
    regions_removed = []
    cid = fig.canvas.mpl_connect('button_press_event',
                                 change_object_on_click(ax, regions_removed,
                                                        regions))

    return fig, ax, regions_removed


def draw_rectangles(ax, regions, bbox_border=5):
    # draw rectangles around segmented regions
    # Define the border between the detected leafs and the
    # red bbox drawn around them
    for region in regions:
        min_row, min_col, max_row, max_col = region.bbox
        rect = mpatches.Rectangle((min_col - bbox_border,
                                   min_row - bbox_border),
                                  max_col - min_col + 2 * bbox_border,
                                  max_row - min_row + 2 * bbox_border,
                                  fill=False, edgecolor='red', linewidth=1)
        ax.add_patch(rect)
        ax.annotate(region.index + 1, (min_col, min_row - 20), color='gray')


def change_object_on_click(ax, regions_removed, regions_sorted):
    """
    Create and return `onclick(event)` event function to be used for
    mpl_connect.

    Two lists are taken. One list, where removed regions are referenced
    (`regions_removed`), and a list, where the sorted regions should be in
    ('regions_sorted').

    Parameters
    ----------
    regions_removed : list of removed regions
    regions_sorted : sorted regions

    Returns
    -------
    function
        A function taking one parameter event, which can be used for
        mpl_connect.
    """
    def move_from_and_reindex(a, b, event):
        for i, r in reversed(list(enumerate(a))):
            min_row, min_col, max_row, max_col = r.bbox
            if (event.xdata >= min_col and event.xdata <= max_col
                    and event.ydata >= min_row and event.ydata <= max_row):
                # Event was in bbox of element, move element from a to b
                region = a.pop(i)
                b.append(region)
                if event.button == 1:
                    # label sorted leafs
                    ax.annotate(len(b), (min_col, min_row - 20), color='red')
                    region.index = len(b) - 1
                if event.button == 3:
                    del region.index
                    # Delete annotation with label i + 1 and replace
                    # annotations with label > i + 1 with label minus one
                    for i_a, annotation in reversed(list(enumerate(ax.texts))):
                        label = annotation.get_text()
                        if int(label) == i + 1:
                            ax.texts.pop(i_a)
                        elif int(label) > i + 1:
                            ax.texts.pop(i_a)
                            new_index = int(label) - 2
                            new_label = str(new_index + 1)
                            a[new_index].index = new_index
                            ax.annotate(new_label,
                                        (annotation._x, annotation._y),
                                        color='gray')

    def onclick(event):
        # (event.button, event.x, event.y, event.xdata, event.ydata)
        if event.button == 1:
            # Left click
            move_from_and_reindex(regions_removed, regions_sorted, event)
        elif event.button == 3:
            # Right click
            move_from_and_reindex(regions_sorted, regions_removed, event)

    return onclick


def save_preferences(preferences, directory, experiment_name,
                     extension='preferences', suffix='.json'):
    name = '{}@{}{}'.format(experiment_name, extension, suffix)
    with open(os.path.join(directory, name), 'w') as pref_file:
        pref_file.write(json.dumps(preferences))


def save_overview_figure(fig, directory, experiment_name, extension='overview',
                         suffix='.jpg', dpi=300, quality=75):
    save_figure(fig, directory, experiment_name, extension=extension,
                suffix=suffix, dpi=dpi, quality=quality)


def save_reorder_figure(fig, directory, experiment_name, extension='reordered',
                        suffix='.jpg', dpi=300, quality=75):
    save_figure(fig, directory, experiment_name, extension=extension,
                suffix=suffix, dpi=dpi, quality=quality)


def save_figure(fig, directory, experiment_name, extension, suffix='.jpg',
                dpi=300, quality=75):
    name = '{}@{}{}'.format(experiment_name, extension, suffix)
    fullname = os.path.join(directory, name)
    fig.savefig(fullname, dpi=dpi, quality=quality)


def save_region_image(image, directory, experiment_name, region,
                      suffix='.png'):
    # Save array as tiff
    name = create_image_name(experiment_name, region)
    filename = '{}{}'.format(name, suffix)
    fullname = os.path.join(directory, filename)
    skimage.io.imsave(fname=fullname, arr=image, optimize=True)


def save_region_images(image, directory, experiment_name, regions,
                       suffix='.png', set_to_zero=True, extra_image_border=0,
                       dpi=300):
    """
    Parameters
    ----------
    set_to_zero : bool
        Set pixels not belonging to the leaf segment to zero
    extra_image_border : int
        Extra pixels around each leaf image
    """
    for i, region in enumerate(regions):
        # Read in coordinates for new image
        min_row, min_col, max_row, max_col = region.bbox

        # Create new image for one leaf and fill with RGB values of the leaf
        if set_to_zero:
            height = max_row - min_row + 2 * extra_image_border
            width = max_col - min_col + 2 * extra_image_border
            img = np.zeros((height, width, 3), dtype='uint8')
            rows = region.coords[:, 0]
            cols = region.coords[:, 1]
            img[rows - min_row + extra_image_border,
                cols - min_col + extra_image_border] = image[rows, cols]
        else:
            idx_y = slice(min_row - extra_image_border,
                          max_row + extra_image_border)
            idx_x = slice(min_col - extra_image_border,
                          max_col + extra_image_border)
            img = image[idx_y, idx_x]

        save_region_image(img, directory, experiment_name, region, suffix)


def autodetect_leafs(image_filename, results_dir, experiment_name,
                     threshold='auto', closing=True, min_area=0,
                     vertical_dilate=0, horizontal_dilate=0, centroid=True,
                     regions_save=True, verbose=False):
    if verbose:
        message = 'Processing image "{}" with experiment name "{}" ...'
        print(message.format(image_filename, experiment_name))
        print('  ', end='')
    image_filename = os.path.abspath(image_filename)
    results_dir = check_dir(results_dir, autocreate=True)
    regions_dir = check_dir(os.path.join(results_dir, 'regions'),
                            autocreate=True)
    images_dir = check_dir(os.path.join(results_dir, 'images'),
                           autocreate=True)
    overview_dir = check_dir(os.path.join(results_dir, 'overview'),
                             autocreate=True)
    preferences_dir = check_dir(os.path.join(results_dir, 'preferences'),
                                autocreate=True)

    # Read image
    image = skimage.io.imread(image_filename)
    # Detect leafs
    mask, image_label = detect_segments(image, threshold, closing=closing)
    # Label image regions
    image_label_overlay = label2rgb(image_label, image=image)

    # List of regions of leafs with an area of at least min_area
    leaf_regions = [region for region in regionprops(image_label)
                    if region.area >= min_area]

    # Automatically group and sort detected leaf regions, according to
    # 1. the rows and 2. the columns
    group_and_sort_regions(leaf_regions, vertical_dilate=vertical_dilate,
                           horizontal_dilate=horizontal_dilate,
                           centroid=centroid, inplace=True, verbose=verbose)

    if verbose:
        print('  Saving results in directory "{}" ...'.format(results_dir))

    # Delete old regions and save autodetected ones
    delete_regions(regions_dir, experiment_name)
    delete_region_images(images_dir, experiment_name)
    delete_reorder_figure(overview_dir, experiment_name)
    if regions_save:
        save_regions(leaf_regions, regions_dir, experiment_name)
    save_region_images(image, images_dir, experiment_name, leaf_regions)

    # Save preferences
    preferences = {
        'image_filename': image_filename,
        'results_dir': results_dir,
        'experiment_name': experiment_name,
        'threshold': threshold,
        'closing': closing,
        'min_area': min_area,
        'vertical_dilate': vertical_dilate,
        'horizontal_dilate': horizontal_dilate,
        'centroid': centroid,
        'regions_save': regions_save
    }

    save_preferences(preferences, preferences_dir, experiment_name)

    if verbose:
        message = '  Saving overview in directory "{}" ...'
        print(message.format(overview_dir), end='')

    # Create overview figure, consisting of:
    # Detected background according to the threshold,
    # the labeled image and the original image
    # Switch off interactive figure
    plt.ioff()
    images = [np.logical_not(mask), image_label_overlay, image]
    fig, axes = create_overview_figure(images, leaf_regions)
    save_overview_figure(fig, overview_dir, experiment_name)

    # Bugfix: Clear figure, to prevent clogging of memory, due to
    # unresolved references from matplotlib figure
    fig.clear()
    plt.close(fig)

    # Switch on interactive figure
    plt.ion()

    if verbose:
        print(' DONE')


def reorder_leafs(image_filename, results_dir, experiment_name):
    image_filename = os.path.abspath(image_filename)
    regions_dir = check_dir(os.path.join(results_dir, 'regions'))

    # Read and show image
    image = skimage.io.imread(image_filename)

    # Read in cached regions
    leaf_regions = read_regions(regions_dir, experiment_name)

    # Switch on interactive figure
    plt.ion()

    # Create reorder figure
    fig, ax, regions_removed = create_reorder_figure(image, leaf_regions)
    fig.show()

    return fig, ax, leaf_regions, regions_removed


def overwrite_with_reordered_leafs(reorder_fig, image_filename, results_dir,
                                   experiment_name, regions,
                                   regions_save=True):
    image_filename = os.path.abspath(image_filename)
    results_dir = check_dir(results_dir, autocreate=True)
    regions_dir = check_dir(os.path.join(results_dir, 'regions'),
                            autocreate=True)
    images_dir = check_dir(os.path.join(results_dir, 'images'),
                           autocreate=True)
    overview_dir = check_dir(os.path.join(results_dir, 'overview'),
                             autocreate=True)

    # Read image
    image = skimage.io.imread(image_filename)

    # Delete old regions and save reordered ones
    delete_regions(regions_dir, experiment_name)
    delete_region_images(images_dir, experiment_name)
    save_region_images(image, images_dir, experiment_name, regions)
    if regions_save:
        save_regions(regions, regions_dir, experiment_name)
    save_reorder_figure(reorder_fig, overview_dir, experiment_name)
