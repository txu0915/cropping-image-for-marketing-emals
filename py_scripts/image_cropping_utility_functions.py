def cropping_out_white_boarder(image_np):
    ## (1) Convert to gray, and threshold
    gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
    th, threshed = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY_INV)
    ## (2) Morph-op to remove noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11,11))
    morphed = cv2.morphologyEx(threshed, cv2.MORPH_CLOSE, kernel)
    ## (3) Find the max-area contour
    cnts = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    cnt = sorted(cnts, key=cv2.contourArea)[-1]
    ## (4) Crop and save it
    x,y,w,h = cv2.boundingRect(cnt)
    dst = image_np[y:y+h, x:x+w]
    return dst

def determine_final_coords(cropped_coords,original_dims,cropped_dims,aspect_ratio = 1, expansion_factor = 0.05):
    w_cropped, h_cropped = cropped_dims
    w, h = original_dims
    w_diff, h_diff = (w-w_cropped)/2.0,(h-h_cropped)/2.0
    left_boud = w_diff
    right_bound = w - w_diff
    top_bound = h_diff
    bottom_bound = h-h_diff
    ## compute shortest dist to original boundaries...
    h_top,w_left,h_bottom,w_right = cropped_coords ## in percentiles...
    w_boundbox = w_right-w_left
    h_boundbox = h_bottom-h_top  ## in percentiles...
    h_center,w_center = (h_top+h_bottom)/2.0, (w_left+w_right)/2.0
    #print("h_top,w_left,h_bottom,w_right",h_top,w_left,h_bottom,w_right)
    if w_boundbox > h_boundbox:
        longer_edge = w_boundbox*w
        new_width = min(int(longer_edge*(1+expansion_factor)),w_cropped)
        new_height = min(int(new_width/aspect_ratio),h_cropped)
        if new_width > new_height*aspect_ratio:
            new_width = new_height*aspect_ratio
        u,b = int(h_top*h), int(h_bottom*h)
        l,r = int(w_left*w),int(w_right*w)
        #l,r = min(int(w_center*w - new_width/2.0),left_boud),max(int(w_center*w + new_width/2.0),right_bound)
        while True:
            if l > left_boud:
                l -= 1
            if r - l >= new_width:
                break
            if r < right_bound and r > 0:
                r += 1
            if r - l >= new_width:
                break  
        while True:
            if u > top_bound:
                u -= 1
            if b - u >= new_height:
                break
            if b < bottom_bound and b > 0:
                b += 1
            if b - u >= new_height:
                break
    else:
        longer_edge = h_boundbox*h
        new_height = min(int(longer_edge*(1+expansion_factor)),h_cropped)
        new_width = min(int(new_height*aspect_ratio),w_cropped)
        if new_width < new_height*aspect_ratio:
            new_height = new_width/aspect_ratio
        #u,b = min(int(h_center*h - new_height/2.0),top_bound), max(int(h_center*h + new_height/2.0),bottom_bound)
        l,r = int(w_left*w),int(w_right*w)
        u,b = int(h_top*h), int(h_bottom*h)
        #print(new_edge_length,"new edge length")
        while True:
            if l > left_boud:
                l -= 1
            if r - l >= new_width:
                break
            if r < right_bound and r > 0:
                r += 1
            if r - l >= new_width:
                break 
        while True:
            if u > top_bound:
                u -= 1
            if b - u >= new_height:
                break
            if b < bottom_bound and b > 0:
                b += 1
            if b - u >= new_height:
                break  
        new_cropped_coords = (u,l,b,r)
    #print("u,l,b,r","|",u,l,b,r)
    swapped_cropp_coords = (l,u,r,b)
    return swapped_cropp_coords

def convert_folder_name(folder_name):
    ans = folder_name.split(" ")[0]
    for element in folder_name.split(" ")[1:]:
        ans += "%20"
        ans += element
    ans += "/"
    return ans

### use wget to download urls at scale...
# my_images = {}
# for i, row in enumerate(my_url_list):
#     print(i, row.split('/')[4])
#     if row.split('/')[4] not in my_images.keys():
#         my_images[row.split('/')[4]] = 1
#         wget.download(row,"./image-samples/"+row.split('/')[4]+".jpg")
#         if i == 100:
#             break
#     else:
#         my_images[row.split('/')[4]] += 1