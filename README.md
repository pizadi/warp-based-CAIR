# warp-based-CAIR
A simple warp-based CAIR algorithm

This algorithm uses OpenCV to perform a content-aware resizing operation on an image.

    resize(I: np.ndarray, E: np.ndarray, target_size: (int, int), mode: int) -> np.ndarray
    
    I:            The input image as an ndarray with the shape (i, j, channels)
    E:            The energy map for I as an ndarray with the shape (i, j)
    target_size:  The target size for warping the image, as a tuple of 2 integers
    mode:         The resize mode for CAIR:
                  mode = 0: Directly resize the image to the target size
                  mode > 0: Resizes the image with a fixed aspect ratio so that its dimensions are
                  greater or equal to target, and then performs CAIR.
                  mode < 0: Resizes the image with a fixed aspect ratio so that its dimensions are
                  smaller or equal to target, and then performs CAIR.
