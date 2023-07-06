import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as ndi

from photoshop.blend.methods import multiply, normal
from photoshop.blend.normal import normal_blend
from photoshop.core.dtype import Float, UInt8
from matplotlib import pyplot as plt

from photoshop.ops.transform import expand_as_rgba

# TODO: use normal blend from methods


def lighting(mask: np.ndarray, elevation: Float, angle: Float, depth: Float) -> np.ndarray:
    zero_point = np.sin(elevation)
    light_normal = np.array([
        np.cos(elevation) * -np.cos(angle),
        np.cos(elevation) * np.sin(angle),
        zero_point
    ])

    dz__ = np.ones(mask.shape).astype(Float) * (1.0 / depth)
    # Compute gradients in the x and y directions using Sobel operator
    # noinspection PyUnresolvedReferences
    dx__ = cv2.Sobel(mask, cv2.CV_64F, 1, 0, ksize=3)
    # noinspection PyUnresolvedReferences
    dy__ = cv2.Sobel(mask, cv2.CV_64F, 0, 1, ksize=3)

    surface_normal = np.dstack((dx__, dy__, dz__)).astype(Float)
    # Calculate incident light amount

    # numerator = np.dot(surface_normal, light_normal)
    # denominator = np.linalg.norm(surface_normal, axis=2) * np.linalg.norm(light_normal)
    # incident_light = numerator / denominator
    # return incident_light

    dot_product = np.dot(surface_normal, light_normal)
    incident_light = ((dot_product / np.linalg.norm(surface_normal, axis=2)) - zero_point) / (1.0 - zero_point)
    return incident_light


def apply_lights(hi: np.ndarray, lo: np.ndarray, incident_light: np.ndarray) -> np.ndarray:
    return hi * np.expand_dims(incident_light, 2)
    # if incident_light >= 0:
    #     hi(x, y) = uint8(255 * incidentLight + T(.5));
    # else
    #     m_lo(x, y) = uint8(-255 * incidentLight + T(.5));


def get_work_rgb(rgba: np.ndarray) -> np.ndarray:
    rgba = expand_as_rgba(rgba).astype(Float) / 255.
    rgb, alpha = rgba[:, :, :3], rgba[:, :, 3]
    bg, bg_alpha = np.zeros(rgb.shape, dtype=UInt8), np.zeros(alpha.shape, dtype=UInt8)
    return (normal_blend(rgb, alpha, bg, bg_alpha) * 255).astype(UInt8)


def compute_antialiased_distance_map(mask, supersample_factor=2):
    # Extract the alpha channel and normalize its values to [0, 1]
    alpha = mask / 255.0

    # Upsample the alpha channel
    upsampled_alpha = ndi.zoom(alpha, supersample_factor, order=1)

    # Compute the distance map using the Euclidean metric on the upsampled alpha
    distance_map = ndi.distance_transform_edt(upsampled_alpha, sampling=3)

    # Downsample the distance map to the original image size using averaging
    downsampled_distance_map = ndi.zoom(distance_map, 1 / supersample_factor, order=1)

    # Rescale the distance map to the range [0, 255]
    downsampled_distance_map = (downsampled_distance_map / np.max(downsampled_distance_map)) * 255

    return downsampled_distance_map.astype(np.uint8)


def get_lightning(mask: np.ndarray, light_elevation: Float, light_angle: Float, depth: Float):
    angle = Float(np.radians(light_angle))
    elevation = Float(np.radians(light_elevation))

    incident_light = lighting(
        mask=mask,
        elevation=elevation,
        angle=angle,
        depth=depth
    )
    return incident_light


def get_smoothing(transform: np.ndarray) -> np.ndarray:
    # noinspection PyUnresolvedReferences
    return cv2.boxFilter(cv2.boxFilter(transform, -1, (3, 3)), -1, (3, 3))


def soften_normals(normals, kernel=None) -> np.ndarray:
    if kernel is None:
        kernel = np.ones((3, 3))
    # Soften the surface normals using convolution or filtering
    return ndi.convolve(normals, kernel)


def rescale_array(arr, new_min, new_max):
    old_min = np.min(arr)
    old_max = np.max(arr)
    scaled_arr = np.interp(arr, (old_min, old_max), (new_min, new_max))
    return scaled_arr


if __name__ == '__main__':
    """
    Photoshop's Bevel and Emboss works predictably:
    
    1) Compute a distance transform in a temporary 8-bit single channel image
    
    Chisel uses the Euclidean Distance Transform with a Chamfer metric (3x3, 5x5, or 7x7 
    depending on size). You can use an exact euclidean distance transform if you'd like, 
    I prefer the one from Meijster since it can be made anti-aliased 
    ("A General Algorithm for Computing Distance Transforms in Linear Time", MEIJSTER).
    
    Smooth Bevel uses a Chamfer 5-7-11 distance transform followed by two applications of a 
    box blur, to produce the bump map.
    
    2) Apply bump mapping to the intermediate distance transform image. Blinn's original technique is suitable.
    
    3) For softening you can perform a convolution on the surface normals or you can filter them using a kernel.
    
    4) Using the bump map the surface normals are combined with the global light source to compute the lighting intensity as a value from -1 to 1 where negative values are shadows, positive values are highlights, and the absolute value is the magnitude of the light source.
    
    5) Two 8-bit single channel temporary images are calculated, one from the highlight intensities and the other from the shadows. From there it is is trivial matter to use each masks to tint the layer using a colour, blend mode, and opacity - one mask for the highlights and the other for the shadows.
    """
    # Smooth bevel
    angle_ = 90
    elevation_ = 30
    depth_ = 1.0

    # noinspection PyUnresolvedReferences
    # 1.1 Load image
    image_ = cv2.imread('text.png', cv2.IMREAD_UNCHANGED)
    # 1.2 Compute Gray Scale
    gray_ = cv2.cvtColor(image_, cv2.COLOR_BGRA2GRAY)
    # 1.3 Transform to common RGBA (instead BGRA)
    image_ = cv2.cvtColor(image_, cv2.COLOR_BGRA2RGBA)
    # 1.4 Compute distance transform
    dist_transform = compute_antialiased_distance_map(mask=image_[:, :, 3])
    # 1.5 ...followed by two applications of a box blur, to produce the bump map
    bump_map = get_smoothing(dist_transform)

    lighting_intensity = get_lightning(
        mask=bump_map,
        light_elevation=elevation_,
        light_angle=angle_,
        depth=depth_
    )
    original_lighting_intensity = lighting_intensity.copy()
    softened_normals = soften_normals(lighting_intensity)
    lighting_intensity = rescale_array(
        softened_normals,
        new_min=original_lighting_intensity.min(),
        new_max=original_lighting_intensity.max()
    )

    hi_light_intensity = lighting_intensity.copy()

    # Rescale the array to the range -1 to 1
    lo_lighting_intensity = rescale_array(lighting_intensity, new_min=-1, new_max=1)

    # lighting_intensity = np.clip(lighting_intensity, -1.0, 1.0)
    # lighting_intensity = softened_normals

    # Highlights
    highlight_color_layer = np.ones((*hi_light_intensity.shape, 4), dtype=UInt8)
    highlight_color_layer[:, :, 0] = 255
    highlight_color_layer[:, :, 1] = 255
    highlight_color_layer[:, :, 2] = 255
    highlight_color_layer[:, :, 3] = 255
    highlight_mask = np.zeros(hi_light_intensity.shape, dtype=UInt8)
    highlight_mask[hi_light_intensity >= 0] = hi_light_intensity[hi_light_intensity >= 0] * 255

    # Shadows
    shadows_color_layer = np.ones((*lo_lighting_intensity.shape, 4), dtype=UInt8)
    shadows_color_layer[:, :, 0] = 0
    shadows_color_layer[:, :, 1] = 0
    shadows_color_layer[:, :, 2] = 0
    shadows_color_layer[:, :, 3] = 255
    shadows_mask = np.zeros(lo_lighting_intensity.shape, dtype=UInt8)
    shadows_mask[lo_lighting_intensity < 0] = lo_lighting_intensity[lo_lighting_intensity < 0] * -255

    # Apply Highlights Only
    background = image_.copy()
    foreground = highlight_color_layer.copy()
    foreground[:, :, 3] = highlight_mask
    hi_img_out = normal(foreground, background, opacity=1.0)

    # Apply Shadows Only
    background = image_.copy()
    foreground = shadows_color_layer.copy()
    foreground[:, :, 3] = shadows_mask
    lo_img_out = normal(foreground, background, opacity=1.0)

    # Apply Shadows Upon Highligh Blended
    background = hi_img_out.copy()
    foreground = shadows_color_layer.copy()
    foreground[:, :, 3] = shadows_mask
    img_out = normal(foreground, background, opacity=1.0)

    # Plot results
    fig, axs = plt.subplots(nrows=4, ncols=3)

    # Row 0
    axs[0][0].imshow(image_, cmap='gray')
    axs[0][0].set_title("Original RGBA", fontsize=10)
    axs[0][1].imshow(gray_, cmap='gray')
    axs[0][1].set_title("Gray", fontsize=10)
    axs[0][2].imshow(bump_map, cmap='gray')
    axs[0][2].set_title("Bump Map", fontsize=10)

    # Row 1
    axs[1][0].imshow(original_lighting_intensity, cmap='gray')
    axs[1][0].set_title("Light Intensity", fontsize=10)
    axs[1][1].imshow(highlight_color_layer, cmap='gray')
    axs[1][1].set_title("Hi Color", fontsize=10)
    axs[1][2].imshow(shadows_color_layer, cmap='gray')
    axs[1][2].set_title("Lo Color", fontsize=10)

    # Row 2
    axs[2][0].imshow(highlight_mask, cmap='gray')
    axs[2][0].set_title("Hi Mask", fontsize=10)
    axs[2][1].imshow(shadows_mask, cmap='gray')
    axs[2][1].set_title("Lo Mask", fontsize=10)
    axs[2][2].imshow(hi_img_out)
    axs[2][2].set_title("Hi Normal Blend", fontsize=10)

    # Row 3
    axs[3][0].imshow(lo_img_out)
    axs[3][0].set_title("Lo Normal Blend", fontsize=10)
    axs[3][1].imshow(img_out)
    axs[3][1].set_title("Result", fontsize=10)

    # axs[3][1].imshow(lo_img_out)
    # axs[3][2].set_title("Lo Blend Normal", fontsize=10)

    # axs[3][0].imshow(shadows_mask, cmap='gray')
    # axs[2][1].imshow(background)
    # axs[2][2].imshow(img_out)

    # [ax.set_axis_off() for ax in axs.ravel()]
    [ax.tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)
     for ax in axs.ravel()]
    plt.axis('off')
    plt.show()
