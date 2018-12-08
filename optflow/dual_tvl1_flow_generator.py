import cv2

# find the optical flow using the tvl1 opencv method and the
def compute_optical_flow_tvl1_opencv(img1,img2,val=False):
    # img_0 and img_1 refers to the first frame and the next frame
    img_0 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img_1 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    if val:
        optical_flow = cv2.DualTVL1OpticalFlow_create(0.25, 0.15, 0.3, 5, 5, 0.01, 30, 10, 0.8, 0.0, 5, False)
    else:
        optical_flow = cv2.DualTVL1OpticalFlow_create(0.25, 0.15, 0.3, 5, 5, 0.01, 10, 5, 0.8, 0.0, 5, False)
    flow = optical_flow.calc(img_0, img_1, None)
    return flow