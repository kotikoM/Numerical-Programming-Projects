import cv2
import numpy as np
import matplotlib.pyplot as plt


def nearest_neighbor_interpolation(input_image, scale_factor):
    # original_height, original_width = input_image.shape[:2]
    # new_width = int(original_width * scale_factor)
    # new_height = int(original_height * scale_factor)
    # return cv2.resize(input_image, (new_width, new_height), interpolation=cv2.INTER_NEAREST)
    original_height, original_width, num_channels = input_image.shape

    new_height = int(original_height * scale_factor)
    new_width = int(original_width * scale_factor)

    output_image = np.zeros((new_height, new_width, num_channels), dtype=input_image.dtype)

    for i in range(new_height):
        for j in range(new_width):
            # Find the nearest pixel from the original image for each channel
            original_i = min(int(i / scale_factor), original_height - 1)
            original_j = min(int(j / scale_factor), original_width - 1)

            output_image[i, j] = input_image[original_i, original_j]

    return output_image


def cubic_interpolate(p, x):
    # Cubic interpolation formula using 4 points
    return (
            p[1] +
            0.5 * x * (p[2] - p[0] +
                       x * (2.0 * p[0] - 5.0 * p[1] + 4.0 * p[2] - p[3] +
                            x * (3.0 * (p[1] - p[2]) + p[3] - p[0])))
    )


def bicubic_interpolation(image, scale):
    # original_height, original_width = image.shape[:2]
    # new_width = int(original_width * scale)
    # new_height = int(original_height * scale)
    # return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    orig_height, orig_width, channels = image.shape
    new_height = int(orig_height * scale)
    new_width = int(orig_width * scale)

    # Output image
    new_image = np.zeros((new_height, new_width, channels), dtype=np.uint8)

    # Scale ratios
    row_scale = orig_height / new_height
    col_scale = orig_width / new_width

    for i in range(new_height):
        for j in range(new_width):
            # Corresponding position in the original image
            x = i * row_scale
            y = j * col_scale

            # Calculate indices of the top-left corner of the 4x4 neighborhood
            x0 = int(np.floor(x)) - 1
            y0 = int(np.floor(y)) - 1

            # Distance from the original pixel position
            dx = x - np.floor(x)
            dy = y - np.floor(y)

            # Iterate over channels (e.g., RGB)
            for c in range(channels):
                # 4x4 neighborhood
                neighborhood = np.zeros((4, 4), dtype=np.float32)

                for m in range(4):
                    for n in range(4):
                        # Get the pixel values, handling boundary conditions with clamping
                        xn = np.clip(x0 + m, 0, orig_height - 1)
                        yn = np.clip(y0 + n, 0, orig_width - 1)
                        neighborhood[m, n] = image[int(xn), int(yn), c]

                # Interpolate along x for each row
                interpolated_rows = np.array([cubic_interpolate(neighborhood[m, :], dy) for m in range(4)])

                # Interpolate along y to get the final value
                pixel_value = cubic_interpolate(interpolated_rows, dx)

                # Clip the value to the range [0, 255] and assign it to the new image
                new_image[i, j, c] = np.clip(pixel_value, 0, 255)

    return new_image


def calculate_error(original_image, resized_image):
    # Ensure the images are the same shape
    if original_image.shape != resized_image.shape:
        raise ValueError("Original and resized images must have the same dimensions.")

    # Calculate the error matrix
    error_matrix = np.abs(original_image - resized_image)

    # Calculate matrix norms
    l1_norm = np.sum(error_matrix)  # L1 norm
    l2_norm = np.sqrt(np.sum(error_matrix ** 2))  # L2 norm
    infinity_norm = np.max(error_matrix)  # Infinity norm

    return l1_norm, l2_norm, infinity_norm


def plot():
    # Convert BGR to RGB for displaying with matplotlib
    input_image_rgb = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
    resized_image_rgb = cv2.cvtColor(resized_image_neighbor, cv2.COLOR_BGR2RGB)
    resized_image_quad_rgb = cv2.cvtColor(resized_image_biquad, cv2.COLOR_BGR2RGB)
    # Enable interactive mode
    plt.ion()
    # Display the original image in a separate window
    plt.figure(figsize=(6, 6))
    plt.imshow(input_image_rgb)
    plt.axis('off')
    plt.show()
    # Display the nearest neighbor resized image in a separate window
    plt.figure(figsize=(6, 6))
    plt.imshow(resized_image_rgb)
    plt.axis('off')
    plt.show()
    # Display the bicubic resized image in a separate window
    plt.figure(figsize=(6, 6))
    plt.imshow(resized_image_quad_rgb)
    plt.axis('off')
    plt.show()
    # Keep the plots open
    plt.ioff()  # Turn off interactive mode
    plt.show()  # This ensures that the last window stays open until closed


if __name__ == '__main__':
    eight = 'images/8x8.jpg'
    tsunami = 'images/tsunami.jpg'
    nike = 'images/nike.png'

    input_image = cv2.imread(tsunami)
    input_image = cv2.resize(input_image, (3840, 2580))

    scale_factor = 5
    # Print dimensions of the input image
    print(f"Input Image Dimensions: {input_image.shape[1]}x{input_image.shape[0]} (Width x Height)")

    resized_image_neighbor = nearest_neighbor_interpolation(input_image, scale_factor)
    #resized_image_neighbor = nearest_neighbor_interpolation(nearest_neighbor_interpolation(input_image, (1/scale_factor)), scale_factor)
    print("Finished resizing using nearest neighbor interpolation")
    resized_image_biquad = bicubic_interpolation(input_image, scale_factor)
    #resized_image_biquad = bicubic_interpolation(bicubic_interpolation(input_image, (1/scale_factor)), scale_factor)
    print(f"Resized Image Dimensions: {resized_image_neighbor.shape[1]}x{resized_image_neighbor.shape[0]} (Width x Height)")

    # original_height, original_width = resized_image_neighbor.shape[:2]
    # new_width = int(original_width)
    # new_height = int(original_height)
    # l1, l2, infinity = calculate_error(input_image, resized_image_neighbor)
    # print("L1 Norm:", l1)
    # print("L2 Norm:", l2)
    # print("Infinity Norm:", infinity)
    #
    # l1, l2, infinity = calculate_error(input_image, resized_image_biquad)
    # print("L1 Norm:", l1)
    # print("L2 Norm:", l2)
    # print("Infinity Norm:", infinity)

    plot()
