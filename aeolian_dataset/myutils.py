import numpy as np

def nan_average(vals, wts):
    if np.all(np.isnan(wts)):
        return np.nan
    else:
        vals = np.array(vals)
        wts = np.array(wts)

        nan_idx = np.isnan(vals) | np.isnan(wts)

        return np.average(vals[~nan_idx], weights = wts[~nan_idx])

def weighted_std(vals, wts):
    if np.all(np.isnan(wts)):
        return np.nan
    else:
        wtd_avg = nan_average(vals, wts = wts)
        return np.sqrt(nan_average((vals - wtd_avg)**2, wts = wts))

def angle_mean(angles_in_rad, wts):
    cosmean = nan_average(np.cos(angles_in_rad), wts = wts)
    sinmean = nan_average(np.sin(angles_in_rad), wts = wts)

    return np.arctan2(sinmean, cosmean)

def angle_median(angles_in_rad):
    cosmean = np.nanmedian(np.cos(angles_in_rad))
    sinmean = np.nanmedian(np.sin(angles_in_rad))

    return np.arctan2(sinmean, cosmean)

def normalize_image(image):
    imin = np.min(image[:,:,0])
    imax = np.max(image[:,:,0])

    image = np.int16(((image - imin)/(imax - imin)) * 255)
    return image

def get_ax(rows=1, cols=1, size=16):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.

    Adjust the size attribute to control how big to render images
    """
    _, ax = pp.subplots(rows, cols, figsize=(size*cols, size*rows))
    return ax

# m = 3.5 since: https://www.itl.nist.gov/div898/handbook/eda/section3/eda35h.htm
# code from https://stackoverflow.com/questions/11686720/is-there-a-numpy-builtin-to-reject-outliers-from-a-list
def reject_outliers(data, m = 2.5):
    d = np.abs(data - angle_median(data))
    mdev = angle_median(d)
    s = d/mdev if mdev else 0.
    return s < m

def area_of_polygon(x, y):
    """Calculates the area of an arbitrary polygon given its verticies"""
    area = 0.0
    for i in range(-1, len(x)-1):
        area += x[i] * (y[i+1] - y[i-1])
    return np.abs(area) / 2.0

def quad_area(lat1, lon1, lat2, lon2, radius):
    m = Basemap(projection='cea',
                llcrnrlat=lat1, llcrnrlon = lon1,
                urcrnrlat = lat2, urcrnrlon = lon2,
                rsphere = radius)

    lats = (lat1, lat1, lat2, lat2, lat1)
    lons = (lon1, lon2, lon2, lon1, lon1)

    xs, ys = m(lons, lats)

    return np.abs(0.5*np.sum(ys[:-1]*np.diff(xs)-xs[:-1]*np.diff(ys)))

def wrapTo180(angle):
    angle = np.radians(angle)
    return np.degrees(np.arctan2(np.sin(angle), np.cos(angle)))

def find_nearest(array, val):
    '''
    Find the value closest to the input argument 'val' in 'array'
    '''
    array = np.asarray(array)
    idx = np.argmin(np.abs(array - val))
    return idx

def get_images_in_bin(in_lon, in_lat, lon, lat, bin_idxs_lon, bin_idxs_lat):
    # Get the closest dune to the input coordinates
    nearest_lon_idx = find_nearest(lon, in_lon)
    nearest_lat_idx = find_nearest(lat, in_lat)

    # Use that dune to get the corresponding bin index
    bin_idx_lon = bin_idxs_lon[nearest_lon_idx]
    bin_idx_lat = bin_idxs_lat[nearest_lat_idx]

    # Get all dunes that correspond to that bin index:
    coords_idxs = np.where((bin_idxs_lat == bin_idx_lat) & (bin_idxs_lon == bin_idx_lon))

    return coords_idxs

def draw_results(res):
    img = res[0][0].copy()
    min_lon, max_lon, min_lat, max_lat = res[0][1:]
    p = res[1]

    cnt = get_contours(img, p,
                    only_barchans = True, confidence = 0.6,
                    is_draw_contour = True, cnt_color = (255, 255, 255))

    get_ax(1, size = 10)
    plt.imshow(img,cmap = 'gray')
    plt.show()

def plot_albedo(image):
    plt.close('all')
    u,v = np.gradient(np.double(cropped))
    ax = gridspec.GridSpec(2, 2, width_ratios=[3, 1], height_ratios = [3, 1])

    plt.subplot(ax[0])
    plt.imshow(cropped)
    plt.xlabel('Length')
    plt.ylabel('Length')
    plt.gca().axis('auto')
    plt.gca().get_xaxis().set_visible(False)
    plt.gca().get_yaxis().set_visible(False)

    # Plot gradient
    gx, gy = get_dune_albedo_gradient(image)

    plt.arrow(int(np.shape(buff)[1]/2),
              int(np.shape(buff)[0]/2),
              -np.mean(gx) * len(buff),
              -np.mean(gy) * len(buff),
              color='w',head_width=1)

    plt.subplot(ax[1])
    l = int(len(cropped[:,20])/2)
    plt.plot(cropped[:,l], np.arange(0, len(cropped[:,20])))
    plt.gca().invert_yaxis()
    plt.gca().yaxis.tick_right()
    plt.gca().yaxis.set_label_position("right")
    plt.xlabel('Brightness')
    plt.ylabel('Length')

    plt.subplot(ax[2])
    plt.plot(cropped[l,:])

    plt.xlabel('Length')
    plt.ylabel('Brightness')
    plt.tight_layout()

    plt.show()

def closest_node(node, nodes):
    nodes = np.asarray(nodes)
    dist_2 = np.sum((nodes - node)**2, axis=1)
    return np.argmin(dist_2)

def is_dune_field_inside_crater(dune_lat, dune_lon, craters_lat, craters_lon,
                                craters_radius_lat, craters_radius_lon, craters_depths, craters_diameters):
    log_idx = (np.abs(craters_lat - dune_lat) < craters_radius_lat) & (np.abs(craters_lon - dune_lon) < craters_radius_lon)
    return (np.any(log_idx), craters_depths[log_idx] / craters_diameters[log_idx], craters_diameters[log_idx])

# A circualr histogram function I found online:
# https://stackoverflow.com/questions/22562364/circular-polar-histogram-in-python
def circular_hist(ax, x, bins=16, density=True, offset=0, gaps=True, face_color = 'b'):
    """
    Produce a circular histogram of angles on ax.

    Parameters
    ----------
    ax : matplotlib.axes._subplots.PolarAxesSubplot
        axis instance created with subplot_kw=dict(projection='polar').

    x : array
        Angles to plot, expected in units of radians.

    bins : int, optional
        Defines the number of equal-width bins in the range. The default is 16.

    density : bool, optional
        If True plot frequency proportional to area. If False plot frequency
        proportional to radius. The default is True.

    offset : float, optional
        Sets the offset for the location of the 0 direction in units of
        radians. The default is 0.

    gaps : bool, optional
        Whether to allow gaps between bins. When gaps = False the bins are
        forced to partition the entire [-pi, pi] range. The default is True.

    Returns
    -------
    n : array or list of arrays
        The number of values in each bin.

    bins : array
        The edges of the bins.

    patches : `.BarContainer` or list of a single `.Polygon`
        Container of individual artists used to create the histogram
        or list of such containers if there are multiple input datasets.
    """
    # Wrap angles to [-pi, pi)
    x = (x+np.pi) % (2*np.pi) - np.pi

    # Force bins to partition entire circle
    if not gaps:
        bins = np.linspace(-np.pi, np.pi, num=bins+1)

    # Bin data and record counts
    n, bins = np.histogram(x, bins=bins)

    # Compute width of each bin
    widths = np.diff(bins)

    # By default plot frequency proportional to area
    if density:
        # Area to assign each bin
        area = n / x.size
        # Calculate corresponding bin radius
        radius = (area/np.pi) ** .5
    # Otherwise plot frequency proportional to radius
    else:
        radius = n

    # Plot data on ax
    patches = ax.bar(bins[:-1], radius, zorder=1, align='edge', width=widths,
                     color=face_color, fill=True, linewidth=1, alpha = 0.5)

    # Set the direction of the zero angle
    ax.set_theta_offset(offset)

    # Remove ylabels for area plots (they are mostly obstructive)
    if density:
        ax.set_yticklabels([])

    return n, bins, patches
