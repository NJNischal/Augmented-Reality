import cv2
import numpy as np 

"""
@brief: Function to select the tag type to be processsed
@param None
@return: Returns the video file chosen
"""
def choice():
    print("Enter the video number(1/2/3):  ")
    print('\n 1 : Tag 0 \n 2 : Tag 1 \n 3 : Tag 2 4 : ')
    selection = int(input())
    if selection == 1:
        video = cv2.VideoCapture('Video/Tag0.mp4')
    elif selection == 2:
        video = cv2.VideoCapture('Video/Tag1.mp4')
    elif selection == 3:
        video = cv2.VideoCapture('Video/Tag2.mp4')
    elif selection == 4:
        video = cv2.VideoCapture('Video/multipleTags.mp4')
    else:
        # print("Invalid Selection Try again", selection)
        raise ValueError("Invalid Selection restart code again")
    return video
        
"""
@brief : Function to get the H and H_inverse matrix from the image
@param : position_value -> The position of the tag with respect reference tag
         x_coord -> The coordinates of the camera,
         y_coord -> The coordinates of tthe camera.
@return : H -> The H matrix
          H_inv -> The inverse H matrix
"""
def homography(position_value, x_coord, y_coord):
    """
    1.) x_world_val : The x value of the world coordinates
    2.) y_world_val : The y value of the world coordinates 
    3.) x_camera_coord : The x value of the camera coordinates
    4.) y_camera_coord : The y value of the camera coordinates
    """
    # Setting the world coordinates
    if position_value == 'belowRight':
        x_world_val1, y_world_value1 = coordinates[0][0][0], coordinates[0][0][1]
        x_world_val2, y_world_value2 = coordinates[1][0][0], coordinates[1][0][1]
        x_world_val3, y_world_value3 = coordinates[2][0][0], coordinates[2][0][1]
        x_world_val4, y_world_value4 = coordinates[3][0][0], coordinates[3][0][1]
    elif position_value == 'belowLeft':
        x_world_val1, y_world_value1 = coordinates[1][0][0], coordinates[1][0][1]
        x_world_val2, y_world_value2 = coordinates[2][0][0], coordinates[2][0][1]
        x_world_val3, y_world_value3 = coordinates[3][0][0], coordinates[3][0][1]
        x_world_val4, y_world_value4 = coordinates[0][0][0], coordinates[0][0][1]
    elif position_value == 'topLeft':
        x_world_val1, y_world_value1 = coordinates[2][0][0], coordinates[2][0][1]
        x_world_val2, y_world_value2 = coordinates[3][0][0], coordinates[3][0][1]
        x_world_val3, y_world_value3 = coordinates[0][0][0], coordinates[0][0][1]
        x_world_val4, y_world_value4 = coordinates[1][0][0], coordinates[1][0][1]
    elif position_value == 'topRight':
        x_world_val1, y_world_value1 = coordinates[3][0][0], coordinates[3][0][1]
        x_world_val2, y_world_value2 = coordinates[0][0][0], coordinates[0][0][1]
        x_world_val3, y_world_value3 = coordinates[1][0][0], coordinates[1][0][1]
        x_world_val4, y_world_value4 = coordinates[2][0][0], coordinates[2][0][1]
    else:
        raise ValueError("Invalid corner provided please recheck !!!!")

    # Setting the camera  coordinates
    x_camera_coords1, y_camera_coords1 = 0, 0
    x_camera_coords2, y_camera_coords2 = x_coord, 0
    x_camera_coords3, y_camera_coords3 = x_coord, y_coord
    x_camera_coords4, y_camera_coords4 = 0, y_coord

    # Calculating the A matrix for calculating the SVD.
    A = [[x_world_val1, y_world_value1, 1, 0, 0, 0, -x_camera_coords1*x_world_val1, -x_camera_coords1*y_world_value1, -x_camera_coords1],
         [0, 0, 0, x_world_val1, y_world_value1, 1, -y_camera_coords1*x_world_val1, -y_camera_coords1*y_world_value1, -y_camera_coords1],
         [x_world_val2, y_world_value2, 1, 0, 0, 0, -x_camera_coords2*x_world_val2, -x_camera_coords2*y_world_value2, -x_camera_coords2],
         [0, 0, 0, x_world_val2, y_world_value2, 1, -y_camera_coords2*x_world_val2, -y_camera_coords2*y_world_value2, -y_camera_coords2],
         [x_world_val3, y_world_value3, 1, 0, 0, 0, -x_camera_coords3*x_world_val3, -x_camera_coords3*y_world_value3, -x_camera_coords3],
         [0, 0, 0, x_world_val3, y_world_value3, 1, -y_camera_coords3*x_world_val3, -y_camera_coords3*y_world_value3, -y_camera_coords3],
         [x_world_val4, y_world_value4, 1, 0, 0, 0, -x_camera_coords4*x_world_val4, -x_camera_coords4*y_world_value4, -x_camera_coords4],
         [0, 0, 0, x_world_val4, y_world_value4, 1, -y_camera_coords4*x_world_val4, -y_camera_coords4*y_world_value4, -y_camera_coords4]]
    # Calculating the  SVD matrix for the homography matrix 
    U, S, Vt = np.linalg.svd(A, full_matrices= True)
    H = np.array(Vt[8, :]/Vt[8, 8]).reshape((3,3))
    # Calculating the inverse of the homography matrix
    H_inv = np.linalg.inv(H)
    return H, H_inv

"""
@brief : Funttion to warp the image from or onto the image
@param : y_coord -> Selected y coordinates to warp
         x_coord -> Selected x coordinates to warp
         H_mat -> The homography matrix
         image -> The input image for each Video frame image
@return : warp_image -> The warped image
"""
def warp_function(y_coord, x_coord, H_mat, image):
    """
    1.) image_height : The height of the input video image 
    2.) image_width : The width of the input video image
    3.) new_x_coord : New calculated x of warp coordinates
    4.) new_y_coord : New calculated y of warp coordinates
    """
    image_height = image.shape[0]
    image_width = image.shape[1]
    updated_coord = np.matmul(H_mat, np.array([[x_coord], [y_coord], [1.0]]))
    new_x_coord = int(updated_coord[1][0] / updated_coord[2][0])
    new_y_coord = int(updated_coord[0][0] / updated_coord[2][0])
    # Condition to check if the selected new coordinates arre within the frame
    if 0 <= new_x_coord < image_height and 0 <= new_y_coord < image_width:
        return image[new_x_coord, new_y_coord]
    return [0, 0, 0]

"""
@brief : Function to select and warp the image from the Video frame
@param : image -> The input image from the each Video frame
         H_mat -> The homogenous matrrix oif the image
         coord -> The x and y coordinates of the image to be warped out 
@return : updated_image -> The extrracted and warped image from the main
          frame.
"""
def warp_out(image, H_mat, coord):
    """
    1.) width,height : The coordinated for the image
    """
    y_coord, x_coord = coord
    updated_image = np.array([[warp_function(i,j,H_mat,image) 
                               for j in range(0,x_coord,12)]
                             for i in range(0,y_coord,12)])
    return updated_image

"""
@brief : Function to warp an image into another image
@param : image -> The input image from each video frame
         H_mat -> The homography matrix for the image
         coord -> The x and y coordinates of the image to be warped in 
         corners -> The corners of the image
@return : new_image -> The updated image with the neew image overlay
"""
def warp_in(image,H_mat,coord,corners):
    """
    1.) min_width : The lower bound of the width in the corners
    2.) min_height : The lower bound height of the height in the corners
    3.) max_width : The upper bound of the width in the corners 
    4.) max_height : The upper boundd of the height in the corners
    """
    new_image = np.zeros((coord[0],coord[1],3), dtype=np.uint8)
    new_image = cv2.drawContours(new_image, [corners], 0, (255, 255, 255), thickness = -1)
    min_width = np.min([corner[0][0] for corner in corners])
    max_width = np.max([corner[0][0] for corner in corners])
    min_height = np.min([corner[0][1] for corner in corners])
    max_height = np.max([corner[0][1] for corner in corners])
#         vector_value = np.vectorize(warp_function, otypes=[np.ndarray],
#                                    excluded=[3, 2, 'H_mat','image'],
#                                     cache=True)
    vector_value = np.vectorize(warp_function, otypes=[np.ndarray],
                               excluded=[2, 3, 'H_mat','image'],
                                cache=True)

    # overlay_image = vector_value(np.array([[y_vector for x_vector in range(min_width, max_width)] for y_vector in range(min_height, max_height)]),
                                # np.array([[x_vector for x_vector in range(min_width, max_width)] for y_vector in range(min_height, max_height)]),
                                # H_mat = H_mat, image = image)
    overlay_image = vector_value(np.array([[y_vector for x_vector in range(min_width, max_width)] for y_vector in range(min_height, max_height)]),
                                 np.array([[x_vector for x_vector in range(min_width, max_width)] for y_vector in range(min_height, max_height)]),
                           H_mat = H_mat, image = image)
    # print("height", min_height, max_height)
    # print("width", min_width, max_width)
    new_image[min_height:max_height,min_width:max_width] = np.array([[i for i in row] for row in overlay_image],
                                                                   dtype=np.uint8)
#         cv2.imshow('warp in', new_image)
    return new_image

"""
@brief : Function to check if the detected image is the tag ot not
@param : image -> The selected image to be warped.
@return : image_border -> The value of the size of the image border
inrtensity if more than 0.90.
"""
def double_check(image):
    # Drawing a recttangle on the tag and checking the outline
    image_outline = cv2.rectangle(image, (10, 10), (70, 70), 0, cv2.FILLED)
    image_border = np.sum(image_outline) / ((80 * 80) * ((64 - 36) / 64)) / 255
    return image_border > 0.92

"""
@brief : Function to get the orientation of the tag by taking the
inner white cells
@param : image -> The input image from the Video Input frame
@return : The orientation of the corner in the image
"""
def orientation(image):
    corners = [("belowRight", np.sum(image[250:300, 250:300]) / 2500),
              ("belowLeft", np.sum(image[250:300, 100:150]) / 2500),
              ("topLeft", np.sum(image[100:150, 100:150 ]) / 2500),
              ("topRight", np.sum(image[100:150, 250:300]) / 2500)]
    # Sort the tuples in the descending order
    corners.sort(key=lambda tup: tup[1], reverse=True)
    # return the largest value from the tuple in the corner tuple
    return corners[0][0]

"""
@brief : Funtion to the get the ID of the tag.
@param : Image -> The input image from the video frame.
@return : The tag Id of the tag.
"""
def get_TagID(image):
    keys = ['TOPL', 'TOPR', 'BOTR', 'BOTL']
    position = { 'BOTL' : [200, 250, 150, 200],
               'BOTR' : [200, 250, 200, 250],
               'TOPR' : [150, 200, 200, 250],
               'TOPL' : [150, 200, 150, 200]}
    tag_id = "".join("1" if np.sum(image[position[keys[o]][0]:position[keys[o]][1],
                                           position[keys[o]][2]:position[keys[o]][3]]) / 2500 > 216
                              else '0' for o in range(0, 4))
    # Assigning the tag numbers to an understanable format          
    if tag_id == 1111:
        tag_id = 1
    elif tag_id == 1110:
        tag_id = 2
    elif tag_id == 1011:
        tag_id = 3
    return tag_id

"""
@brief : Function to generate the matrix cube on the image
@param : H_inv -> The inverse matrix for the homography matrix
@return : The rotation and transverse matrix
"""
def generate_matrix_cube(H_inv):
    k_matrix = np.transpose(np.array([[1406.084154, 0, 0],
                        [2.206797, 1417.999306, 0],
                        [1014.136434, 566.347754, 1]]))
    k_inv_matrix  = np.linalg.inv(k_matrix)
    b_matrix = np.matmul(k_inv_matrix,H_inv)
    b1 = b_matrix[:, 0].reshape(3, 1)
    b2 = b_matrix[:, 1].reshape(3, 1)
    b3 = b_matrix[:, 2].reshape(3, 1)

    scalar_value = 2 / (np.linalg.norm(np.dot(k_inv_matrix,b1))+
                       np.linalg.norm(np.dot(k_inv_matrix,b2)))
    # Calculating the transverse matrix
    trans_matrix = scalar_value * b3
    # Calculating the rotational matrix
    rot_mat1 = scalar_value * b1
    rot_mat2 = scalar_value * b2
    rot_mat3 = ((np.cross(b_matrix[:,0],b_matrix[:,1])) * scalar_value * scalar_value).reshape(3, 1)
    rot_matrix = np.concatenate((rot_mat1, rot_mat2, rot_mat3), axis = 1)
    return rot_matrix, trans_matrix, k_matrix

"""
@brief : Function to draw the 3d cube on the tag
@param : image-> The input frame from the video
         points -> The 3D points to draw the cube
@return : The image frame with the overlayed cube
"""
def draw_cube(image, points):
    points = np.int32(points).reshape(-1, 2)
    # draw Coordinates of the ground plane
    image = cv2.drawContours(image, [points[:4]], -1, (0, 255, 0),3)
    for i, j in zip(range(4), range(4,8)):
        image = cv2.line(image, tuple(points[i]), tuple(points[j]),
                        (0, 0, 0), 3)
    # Draw Coordinates for the top plane
    image = cv2.drawContours(image, [points[4:]], -1, (255, 255, 0),3)
    return image

image_x_points, image_y_points = 400,400
cube_coordinates = np.float32([[0, 0, 0], [0, 500, 0], [500, 500, 0],
                               [500, 0, 0], [0, 0, -200], [0, 500, -200],
                               [500, 500, -200], [500, 0, -200]])
picture = cv2.imread('Lena.png')
(picture_x, picture_y, ch) = picture.shape
scale = 0.4
video = None
video = choice()

while True:
    val, curr_frame = video.read()
    if val is None or val is False:
        # raise ValueError("The Video was not Imported")
        print("Video is done ")
        break
    curr_frame_orginal = curr_frame.copy()
    rows, cols, chs = curr_frame.shape
    curr_frame_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
    curr_frame_threshold = cv2.threshold(curr_frame_gray, 200,255,0)[1]
    trash_value, contours, heirarchy = cv2.findContours(curr_frame_threshold,
                                                       cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    coordinates_3D = curr_frame
    overlay_result = curr_frame
    tag_matrix = np.zeros((image_y_points, image_x_points), dtype = np.uint8)
    warped_image = np.zeros((curr_frame.shape[0], curr_frame.shape[1], 3), dtype = np.uint8)
    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        coordinates = cv2.approxPolyDP(contour, 0.01 * perimeter, True)
        area = cv2.contourArea(contour)
        # print("The number of coordinates are", len(coordinates))
        if 600 <= area < 24000:
            if len(coordinates) == 4:
                try:
                    H_matrix, H_inv_matrix = homography('belowRight', image_x_points, image_y_points)
#                     H_matrix, H_inv_matrix = homographymatrix('belowRight', image_x_points, image_y_points)
                except np.linalg.LinAlgError:
                    continue
                image_unwarp = cv2.resize(warp_out(curr_frame, H_inv_matrix,(image_x_points,image_y_points)),
                                         dsize = None, fx =12, fy = 12)
                image_unwarp_gray = cv2.cvtColor(image_unwarp, cv2.COLOR_BGR2GRAY)
                image_unwarp_threshold = cv2.threshold(image_unwarp_gray, 200, 255, cv2.THRESH_BINARY)[1]
                ratio_image = cv2.resize(image_unwarp_threshold, dsize = None, fx =0.2, fy =0.2)
                if not double_check(ratio_image):
                    position_ID = orientation(image_unwarp_threshold)
                    pos_dict = {'topLeft': cv2.ROTATE_180,
                               'topRight': cv2.ROTATE_90_CLOCKWISE,
                               'belowLeft' : cv2.ROTATE_90_COUNTERCLOCKWISE}
                    tag_matrix = np.copy(image_unwarp_threshold) if position_ID == 'belowRight' \
                    else cv2.rotate(image_unwarp_threshold, pos_dict[position_ID])
                    tag_id = int(get_TagID(tag_matrix))
                    tagx1, tagy1 = coordinates[0][0][0], coordinates[0][0][1]
                    ## Insert the  text overlay here
                    # cv2.putText(curr_frame_orginal, "Tag_ID : %d" %tag_id, (tagx1-50, tagy1-50), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 225), 2, cv2.LINE_AA)
                    #####
                    try:
                        H_matrix_image, H_inv_matrix_image = homography(position_ID,picture_x, picture_y)
#                         H_matrix_image, H_inv_matrix_image = homographymatrix(position_ID,picture_x, picture_y)
                    except np.linalg.LinAlgError:
                        continue
                    warped_image = np.bitwise_or(warped_image,
                                                 warp_in(picture, H_matrix_image, (rows,cols), coordinates))
#                             cv2.imshow("warped", warped_image)
                    warped_image_gray = cv2.cvtColor(warped_image,cv2.COLOR_BGR2GRAY)
                    warped_image_threshold = cv2.threshold(warped_image_gray, 0, 250, cv2.THRESH_BINARY_INV)[1]
                    curr_frame_slotted = cv2.bitwise_and(curr_frame_orginal, curr_frame_orginal, 
                                                        mask =  warped_image_threshold)
                    overlay_result = cv2.add(curr_frame_slotted, warped_image)
                    # cv2.imshow("overlay", overlay_result)
                    # cv2.imshow("slottted", curr_frame_slotted)
                    rotation, translation, k_matrix = generate_matrix_cube(H_inv_matrix_image)
                    coordinates_3D, jacobian = cv2.projectPoints(cube_coordinates, rotation,
                                                                translation, k_matrix, np.zeros((1, 4)))
                    coordinates_3D = draw_cube(curr_frame, coordinates_3D)
    # Setting the display values and 
    curr_frame_disp = cv2.resize(curr_frame_orginal, dsize=None, fx = scale, fy = scale)
    overlay_result_disp = cv2.resize(overlay_result, dsize=None, fx = scale, fy = scale)
    coordinates_3D_disp = cv2.resize(coordinates_3D, dsize=None, fx= scale, fy = scale)
    image_unwarp_disp = cv2.resize(image_unwarp_threshold, dsize=None, fx = scale, fy = scale)
    horizontal_pad = (curr_frame_disp.shape[1] - tag_matrix.shape[1]) // 2
    vertical_pad = (curr_frame_disp.shape[0] - tag_matrix.shape[0]) // 2
    coordinates_3D_disp = cv2.copyMakeBorder(coordinates_3D_disp,top =0,
                                            bottom =0, left = coordinates_3D_disp.shape[1] // 2,
                                            right = coordinates_3D_disp.shape[1] // 2,
                                            borderType = cv2.BORDER_CONSTANT, value =(48, 48, 48))
    image_unwarp_disp = cv2.copyMakeBorder(image_unwarp_disp,top =0,
                                            bottom =0, left = image_unwarp_disp.shape[1],
                                            right = image_unwarp_disp.shape[0],
                                            borderType = cv2.BORDER_CONSTANT, value =(48, 48, 48))
    final_display = np.concatenate((np.concatenate((curr_frame_disp, overlay_result_disp),axis = 1),
                                  coordinates_3D_disp),
                                  axis = 0)

    cv2.imshow("Result Windows", final_display)
    # cv2.imshow("Warp", image_unwarp_disp)
    key = cv2.waitKey(1)
    if key == 27:
        break
video.release()
cv2.destroyAllWindows()
