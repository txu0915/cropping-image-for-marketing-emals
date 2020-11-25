import numpy as np
import pandas as pd
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
from io import BytesIO
import cv2
import urllib3
import wget
from google.cloud import bigquery
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")
sys.path.append('/home/tianlongxu/Image_Type_Classifier_Proj/')
sys.path.append('/home/tianlongxu/cropping-images/models/research')
sys.path.append('/home/tianlongxu/cropping-images/models/research/slim')
sys.path.append('/home/tianlongxu/cropping-images/models/research/object_detection/')
sys.path.append('/usr/local/python3')
from Model_Mart_Scripts.utility_functions import download_image
from Model_Mart_Scripts.utility_functions import create_dir_if_needed
from py_scripts.image_cropping_utility_functions import *
def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    im_array = np.array(image.getdata())
    num_channels = im_array.shape[1]
    return im_array.reshape((im_height, im_width, num_channels)).astype(np.uint8)

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
    #print("u,l,b,r","|",u,l,b,r)
    ## final check if aspect-ratio is satisfied...
    if aspect_ratio*(b - u) < r - l:
        horiz_center = (l+r)/2.0
        r = int(horiz_center+aspect_ratio*(b - u)/2.0)
        l = int(horiz_center-aspect_ratio*(b - u)/2.0)
    if aspect_ratio*(b - u) > r - l:
        verti_center = (b+u)/2.0
        b = int(verti_center+(r-l)/(aspect_ratio*2.0))
        u = int(verti_center-(r-l)/(aspect_ratio*2.0))
    swapped_cropp_coords = (l,u,r,b)
    return swapped_cropp_coords

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


from object_detection.utils.label_map_util import load_labelmap
from object_detection.utils.label_map_util import convert_label_map_to_categories
from object_detection.utils.label_map_util import create_category_index
from object_detection.utils import visualization_utils as vis_util

prefix = './'

# What model to download.
#MODEL_NAME = 'bathroomdata'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = prefix+ 'alex-models' + '/kitchen_inference_graph.pb'
# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join(prefix + 'alex-models', 'kitchen-detection.pbtxt')



model_dict = {
    "bathroom":{"PATH_TO_CKPT" : prefix+ 'alex-models' + '/bathroom_inference_graph.pb',
               "PATH_TO_LABELS" : os.path.join(prefix + 'alex-models', 'bathroom-detection.pbtxt'),
               "NUM_CLASSES" : 9},
    "diningroom":{"PATH_TO_CKPT" : prefix+ 'alex-models' + '/diningroom_inference_graph.pb',
               "PATH_TO_LABELS" : os.path.join(prefix + 'alex-models', 'diningroom-detection.pbtxt'),
                 "NUM_CLASSES" : 8},
    "kitchen":{"PATH_TO_CKPT" : prefix+ 'alex-models' + '/kitchen_inference_graph.pb',
               "PATH_TO_LABELS" : os.path.join(prefix + 'alex-models', 'kitchen-detection.pbtxt'),
              "NUM_CLASSES" : 11},
    "outdoor":{"PATH_TO_CKPT" : prefix+ 'alex-models' + '/outdoor_inference_graph.pb',
               "PATH_TO_LABELS" : os.path.join(prefix + 'alex-models', 'outdoor-detection.pbtxt'),
              "NUM_CLASSES" : 12},
    "livingroom":{"PATH_TO_CKPT" : prefix+ 'alex-models' + '/livingroom_inference_graph.pb',
               "PATH_TO_LABELS" : os.path.join(prefix + 'alex-models', 'livingroom-detection.pbtxt'),
                 "NUM_CLASSES" : 11},
    "bedroom":{"PATH_TO_CKPT" : prefix+ 'alex-models' + '/bedroom_inference_graph.pb',
               "PATH_TO_LABELS" : os.path.join(prefix + 'alex-models', 'bedroom-detection.pbtxt'),
              "NUM_CLASSES" : 11}
}

for k, v in model_dict.items():
    PATH_TO_CKPT = v['PATH_TO_CKPT']
    PATH_TO_LABELS = v['PATH_TO_LABELS']
    NUM_CLASSES = v["NUM_CLASSES"]
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    label_map = load_labelmap(PATH_TO_LABELS)
    categories = convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
    category_index = create_category_index(categories)
    model_dict[k]['detection_graph'] = detection_graph
    model_dict[k]['categories'] = categories
    model_dict[k]['category_index'] = category_index
print("[Info] OD models compiled successfully")
inverse_model_dict = {}
for m, v in model_dict.items():
    print(m, v['categories'])
    for pair in v['categories']:
        print(pair['name'])
        inverse_model_dict[pair['name']] = m
print("[Info] inverse model dictionary created successfully")
client=bigquery.Client(project='analytics-online-data-sci-thd')

query = """
with popular_goods as (SELECT * EXCEPT(SCORE)
FROM
(
  SELECT
    P.OMS_ID ,
    CLASS,
    SUBCLASS,
    TAXONOMY ,
    SCORE,
    ROW_NUMBER() OVER(PARTITION BY CLASS ORDER BY SCORE DESC) AS ROW_NBR 
  FROM
    (
      SELECT DISTINCT
        OMS.OMS_ID , P.CLASS, P.SUBCLASS  , P.TAXONOMY 
      FROM
        `pr-edw-views-thd.NURO.PRODUCT` P
      INNER JOIN
        `pr-edw-views-thd.NURO.SKU_UPC_BY_OMS` OMS
      ON 
        P.OMS_ID = OMS.OMS_ID
        AND P.SKU_NBR = OMS.SKU_NBR 
        AND P.SKU_CRT_DT = OMS.SKU_CRT_DT 
        AND P.UPC_CD = OMS.UPC_CD 
      ) P
  INNER JOIN
    `analytics-online-thd.Master.top_seller_score` TS
  ON
    P.OMS_ID = TS.OMS_ID
)
WHERE ROW_NBR <= 1000), --top 1k SKUs from each Class BASED ON TOP SELLER SCORE)

mart_taxonomy as (
  select oms_sku omsid, array_reverse(split(taxonomy, ">"))[safe_offset(0)] as leafnode, * except(oms_sku)
  from `mart.lookup_product_taxonomy` 
),
taxonomy_stacked as (
  select omsid, taxonomy, l1 as l_n, leafnode
  from mart_taxonomy
  union all
  select omsid, taxonomy, l2 as l_n, leafnode
  from mart_taxonomy
  union all
  select omsid, taxonomy, l3 as l_n, leafnode
  from mart_taxonomy
  union all
  select omsid, taxonomy, l4 as l_n, leafnode
  from mart_taxonomy
  union all
  select omsid, taxonomy, l5 as l_n, leafnode
  from mart_taxonomy
  union all
  select omsid, taxonomy, l6 as l_n, leafnode
  from mart_taxonomy
)  ,
taxonomy_od_labels_mapping as(
select distinct *
from taxonomy_stacked 
where l_n in ('bath rugs and mats', 'centerset bathroom sink faucets', 'sconces', 'showerheads', 'single handle bathroom sink faucets', 'toilet bowls', 'vanities with tops', 'vanity mirrors', 'widespread bathroom sink faucets', 'accent chairs', 'bookcases', 'chandeliers', 'dining chairs', 'kitchen and dining tables', 'pendant lights', 'sideboards and buffets', 'vases', 'bar stools', 'built-in dishwasher', 'electric or gas range', 'french door refrigerator', 'knife set', 'over the range microwave', 'pull down faucet', 'tea kettle', 'wall mount range hood', 'wall oven', 'market umbrella', 'outdoor bench', 'outdoor chaise lounge', 'outdoor coffee table', 'outdoor dining chair', 'outdoor lounge chair', 'outdoor pillow', 'outdoor rug', 'outdoor side table', 'outdoor sofa', 'patio dining table', 'planter', 'area rugs', 'coffee tables', 'console tables', 'end tables', 'floor lamps', 'ottomans', 'sofas and loveseats', 'table lamps', 'throw pillows', 'bedroom benches', 'beds and headboards', 'dressers and chests', 'flush mount lights', 'nightstands', 'wall mirrors',
'starting from here are corrected categories',
'wall ovens',
'wall mount range hoods',
'bathroom vanities with tops',
'tea kettles',
'sofas & loveseats',
'sideboards & buffets',
'pull down faucets',
'planters',
'patio dining tables',
'over-the-range microwaves',
'sofas',
'outdoor side tables',
'outdoor rugs',
'outdoor pillows',
'outdoor lounge chairs',
'outdoor dining chairs',
'outdoor coffee tables',
'outdoor chaise lounges',
'outdoor benches',
'market umbrellas',
'knife sets',
'kitchen & dining tables',
'french door refrigerators',
'gas range',
'dressers & chests',
'built-in dishwashers',
'kids beds & headboards')),

sampled_skus_of_selected_taxonomies as (
select * from (select ROW_NUMBER() OVER(PARTITION BY taxonomy) AS row_num, *  from taxonomy_od_labels_mapping tm ) 
)--where row_num <= 400) -- limiting to 400 samples per taxonomy

select distinct sampled_skus_of_selected_taxonomies.omsid, sampled_skus_of_selected_taxonomies.taxonomy, l_n od_label, itc.image_url  
from sampled_skus_of_selected_taxonomies join `mart.itc_weekly_results` itc on sampled_skus_of_selected_taxonomies.omsid   = safe_cast(itc.oms_id as string)
where itc.prediction_labels like "%lifestyle%" and taxonomy  is not null 
AND sampled_skus_of_selected_taxonomies.omsid in (select distinct safe_cast(OMS_ID as string) from popular_goods)
order by taxonomy
"""
input_csv = client.query(query).to_dataframe()
print("[Info] data queried and passed to pandas dataframe")
checked = 0
covered_taxonomies = {}
for i, tax in enumerate(input_csv.taxonomy.unique()):
    total_classes = 0
    for c, od_m in inverse_model_dict.items():
        total_classes += 1
        if c.lower() in tax.lower():
            checked+=1
            covered_taxonomies[tax] = {'od_label':c,'model_name':od_m}
            break
    if total_classes == 54:
        print(i,tax)
print(checked)
covered_taxonomies['furniture>living room furniture>sofas'] = {'od_label':'Sofas and Loveseats','model_name':'livingroom'}
covered_taxonomies['furniture>bedroom furniture>dressers & chests'] = {'od_label':'Dressers and Chests','model_name':'bedroom'}
covered_taxonomies['appliances>microwaves>over-the-range microwaves'] ={'od_label':'Over the Range Microwave','model_name':'kitchen'}
covered_taxonomies['furniture>living room furniture>sofas & loveseats'] = {'od_label':'Sofas and Loveseats','model_name':'livingroom'}
covered_taxonomies['furniture>kitchen & dining room furniture>sideboards & buffets'] = {'od_label':'Sideboards and Buffets','model_name':'diningroom'}
covered_taxonomies['furniture>kitchen & dining room furniture>kitchen & dining tables'] = {'od_label':'Kitchen and Dining Tables','model_name':'diningroom'}
covered_taxonomies['furniture>kids & baby furniture>kids furniture>kids bedroom furniture>kids beds & headboards'] = {'od_label':'Beds and Headboards','model_name':'bedroom'}

print("[Info] all possible taxonomies covered successfully")
print("[Info] initializing the OD servers...")


nbr_cropped = 0
results_path = './results/nov-23-outputs/'
for j, row in input_csv.iterrows():
    if j == 0:
        print("[Info] compute job started...")
    img_pth,taxonomy,image_name = row.image_url,row.taxonomy,row.image_url.split("/")[4]
    if type(taxonomy) == str and taxonomy in covered_taxonomies.keys():
        try:
            resp = urllib.request.urlopen(img_pth)
        except Exception:
            continue
        image = np.asarray(bytearray(resp.read()), dtype="uint8")
        image_np = cv2.imdecode(image, cv2.IMREAD_COLOR)
        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        image_np_no_white_boarder = cropping_out_white_boarder(image_np)
        w_cropped, h_cropped = image_np_no_white_boarder.shape[1],image_np_no_white_boarder.shape[0]
        detection_graph = model_dict[covered_taxonomies[taxonomy]['model_name']]['detection_graph']
        target_OD_result = covered_taxonomies[taxonomy]['od_label']
        categories = model_dict[covered_taxonomies[taxonomy]['model_name']]['categories']
        print(j,"| Using the",covered_taxonomies[taxonomy]['model_name'],"model | target label:",target_OD_result)
        with detection_graph.as_default():
            with tf.Session(graph=detection_graph) as sess:
                image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
                detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
                detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
                detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
                num_detections = detection_graph.get_tensor_by_name('num_detections:0')
                (boxes, scores, classes, num) = sess.run(
                [detection_boxes, detection_scores, detection_classes, num_detections],
                feed_dict={image_tensor: np.expand_dims(image_np, axis=0)})
                for i,score in enumerate(np.squeeze(scores)[6::-1]):
                    if np.squeeze(scores)[i] >= 0.6 and categories[np.squeeze(classes)[i].astype(int)-1]['name'] == target_OD_result:
                        nbr_cropped += 1
                        print(j,"of",input_csv.shape[0],"|","image_cropped | Total:", nbr_cropped)
                        for asp in [1.0,1.77]:
                            final_coords = determine_final_coords(boxes[0,i],(image_np.shape[0],image_np.shape[1]),(w_cropped, h_cropped),aspect_ratio=asp, expansion_factor = 1.0)
                            img = Image.fromarray(image_np).crop(final_coords)
                            create_dir_if_needed(results_path+target_OD_result+'/'+str(image_name)+'.jpg')
                            fname = str(row.omsid)+'|'+str(image_name)+'|'+covered_taxonomies[taxonomy]['model_name']+'|'+target_OD_result+'|'+str(asp)+'.jpg'
                            img.save(results_path+target_OD_result+'/'+fname, quality=100)
print("[Info] Job done, all results saved...")







