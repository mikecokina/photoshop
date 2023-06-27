import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as ndi

from photoshop.blend.methods import multiply, normal
from photoshop.blend.normal import normal_blend
from photoshop.core.dtype import Float, UInt8
from matplotlib import pyplot as plt

from photoshop.ops.transform import expand_as_rgba


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

    highlight_color_layer = np.ones((*lighting_intensity.shape, 4), dtype=UInt8)
    highlight_color_layer[:, :, 0] = 255
    highlight_color_layer[:, :, 1] = 255
    highlight_color_layer[:, :, 2] = 255
    highlight_color_layer[:, :, 3] = 255
    highlight_mask = np.zeros(lighting_intensity.shape, dtype=UInt8)
    highlight_mask[lighting_intensity >= 0] = lighting_intensity[lighting_intensity >= 0] * 255

    # softened_normals = soften_normals(lighting_intensity)
    # lighting_intensity = np.clip(softened_normals, -3, 3)
    # # lighting_intensity = np.clip(lighting_intensity, -1, 1)
    #
    # highlight_mask = np.zeros(lighting_intensity.shape, dtype=UInt8)
    # shadow_mask = np.zeros(lighting_intensity.shape, dtype=UInt8)
    #
    # highlight_mask[lighting_intensity >= 0] = lighting_intensity[lighting_intensity >= 0] * 255
    # shadow_mask[lighting_intensity < 0] = lighting_intensity[lighting_intensity < 0] * -255

    # print()
    background = image_.copy()
    foreground = highlight_color_layer.copy()
    foreground[:, :, 3] = highlight_mask

    img_out = normal(foreground, image_, opacity=1.0)

    # highlight_color = np.ones((*gray_.shape, 3), dtype=UInt8)
    # highlight_color[:, :, 0] = 255
    #
    # shadow_color = np.array([0, 0, 1])

    # highlight_layer = tint_layer(image_[:, :, :3], highlight_mask, highlight_color, blend_mode, opacity)
    # shadow_layer = tint_layer(image_, shadow_mask, shadow_color, blend_mode, opacity)

    fig, axs = plt.subplots(nrows=3, ncols=3)
    axs[0][0].imshow(image_, cmap='gray')
    axs[0][1].imshow(gray_, cmap='gray')
    axs[0][2].imshow(dist_transform, cmap='gray')

    axs[1][0].imshow(bump_map, cmap='gray')
    axs[1][1].imshow(original_lighting_intensity, cmap='gray')
    axs[1][2].imshow(highlight_color_layer)

    axs[2][0].imshow(highlight_mask, cmap='gray')
    axs[2][1].imshow(background)
    axs[2][2].imshow(img_out)
    plt.show()

    #
    # # work_rgb = get_work_rgb(image_)
    #
    # # Apply lights to mask hi.
    # mask_hi = np.ones((*gray_.shape, 3), dtype=UInt8)
    # mask_hi[:, :, 1] = 255
    # with_lights = apply_lights(mask_hi, mask_hi, incident_lights)
    #
    # # mask_rgb = np.ones((*gray_.shape, 3), dtype=Float) * np.expand_dims(with_lights, 2)
    #
    # # bg, bg_alpha = np.zeros(image.shape, dtype=UInt8), np.zeros(dist_mask.shape, dtype=UInt8)
    # # image = normal_blend(image, dist_mask, bg, bg_alpha, opacity=0.75)
    #
    # # image = fg_rgb * fg_a * opacity + bg_rgb * bg_a * (1 - fg_a * opacity)
    # image = with_lights * np.expand_dims(gray_, 2)
    #
    # fig, axs = plt.subplots(nrows=1, ncols=3)
    # axs[0].imshow(gray_)
    # axs[1].imshow(with_lights)
    # plt.show()
