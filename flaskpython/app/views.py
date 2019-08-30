from app import app
from app.helpers import get_page_url_name, get_page_display_name
from flask import render_template, request, redirect, url_for, session, flash
import markdown
import os
import base64
import testmodel
import cv2
import numpy as np
import tensorflow as tf
import restapi
import json
import autocorrect
import time
out1 = ""
model = testmodel.get_model()
file_name_true = ""
model.load_weights("sgen_25_15.hdf5")
graph = tf.get_default_graph()
def crop(image):
    trans_mask = image[:,:,3] == 0

    #replace areas of transparency with white and not transparent
    image[trans_mask] = [255, 255, 255, 255]

    img2 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _,image2 =  cv2.threshold(img2,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    #imgtrans = image2
    min_left_pixel = 10000
    max_right_pixel = -1
    left_pixel = 100000
    right_pixel = -1

    min_top_pixel = 10000
    max_bottom_pixel = -1
    top_pixel = 100000
    bottom_pixel = -1
    l = image2.shape[0]
    w  = image2.shape[1]
    print(image2.shape)
    for i in range(0,l):
        flag = 0
        for j in range(0,w):
            if image2[i][j]<150:
                right_pixel = j
                bottom_pixel = i
                if flag == 0:
                    left_pixel = j
                    top_pixel = i
                    flag = 1
        #print(le)
        min_left_pixel = min(min_left_pixel, left_pixel)
        max_right_pixel = max(max_right_pixel , right_pixel)

        min_top_pixel = min(min_top_pixel, top_pixel)
        max_bottom_pixel = max(max_bottom_pixel , bottom_pixel)

    image2 = image2[min_top_pixel : max_bottom_pixel+20 , min_left_pixel : max_right_pixel+20]

    img = image2

    imgSize = (32,128)
    (ht, wt) = imgSize
    (h, w) = img.shape
    fx = w / wt
    fy = h / ht
    f = max(fx, fy)
    newSize = (max(min(wt, int(w / f)), 1), max(min(ht, int(h / f)), 1)) # scale according to f (result at least 1 and at most wt or ht)
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, newSize)
    target = np.ones([ht, wt]) * 255
    target[0:newSize[1], 0:newSize[0]] = img
    img = target
    img = np.stack((img,)*3, axis=-1)

    return img



@app.route('/')
def index():
    return render_template("index.html")

@app.route('/pages')
def pages():
    ensure_pages_directory_exists()
    pages = os.listdir('pages')
    global out1
    return render_template("pages.html", pages=pages,out=out1)

@app.route('/hand')
def hand():
    ensure_hand_directory_exists()
    pages = os.listdir('hand')
    global out1
    return render_template("hand.html", pages=pages,out=out1)


def ensure_pages_directory_exists():
    if not os.path.exists('pages'):
        os.makedirs('pages')

def ensure_hand_directory_exists():
    if not os.path.exists('hand'):
        os.makedirs('hand')

def ensure_online_directory_exists():
    if not os.path.exists('online'):
        os.makedirs('online')

@app.route('/online')
def online():
    ensure_online_directory_exists()
    pages = os.listdir('online')
    global out1
    return render_template("online.html", pages=pages,out=out1)

@app.route('/pages/<page_name>')
def page(page_name):
    with open('pages/' + page_name + '.md') as page_file:
        contents = page_file.read()
    
    html = markdown.markdown(contents)
    return render_template('page.html', page_name=page_name, contents=html)

@app.route('/new_page', methods=['GET', 'POST'])
def new_page():
    if request.method == 'GET':
        return render_template('new_page.html')

    ensure_pages_directory_exists()

    title = request.form['title']
    contents = request.form['contents']

    with open('pages/' + get_page_url_name(title) + '.md', 'w') as page_file:
        page_file.write(contents)

    return redirect(url_for('pages'))

@app.route('/edit_page/<page_name>', methods=['GET', 'POST'])
def edit_page(page_name):
    if request.method == 'GET':
        with open('pages/' + page_name + '.md') as page_file:
            contents = page_file.read()

        title = get_page_display_name(page_name)

        return render_template('edit_page.html', page_name=title, contents=contents)

    title = request.form['title']
    contents = request.form['contents']

    with open('pages/' + get_page_url_name(title) + '.md', 'w') as page_file:
        page_file.write(contents)

    return redirect(url_for('page', page_name=get_page_url_name(title)))

@app.route('/delete_page/<page_name>', methods=['GET', 'POST'])
def delete_page(page_name):
    if request.method == 'GET':
        return render_template('delete_page.html')

    os.remove('./pages/' + page_name + '.md')

    return redirect(url_for('pages'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'GET':
        return render_template('login.html')

    user_name = request.form['username']
    
    #password = request.form['pwd']

    if is_valid_login(user_name):
        session['username'] = user_name

        return redirect(url_for('index'))

    return redirect(url_for('index'))

def is_valid_login(user_name):
    return True

@app.route('/logout', methods=['GET', 'POST'])
def logout():
    if request.method == 'GET':
        return render_template('logout.html')
    
    session.pop('username', None)
    return redirect(url_for('index'))


@app.route('/image', methods=['GET', 'POST'])
def your_view():
    if request.method == 'POST':
        data_url = request.get_data('image')   # here parse the data_url out http://xxxxx/?image={dataURL}
        data_url = data_url.decode("utf-8") 
        
        #print(data_url)

        #print("-================")
        words =  data_url[90:-45].split(";")[2]
        
        
        words = words.split("[")[1] 
        words = words.replace('"',"")
        imdexn = words.index("]")
        #words = words.replace("]","")
        #print(words)
        #print("-================")

        trace = data_url[90:-45].split("[[[")[1]
        trace = trace.split("]]]")[0]
        f = open("./json-file/"+ str(time.time()) +"_"+words[:imdexn]+"_"+ ".json","w+")
        f.write("[[[" + trace + "]]]\n")
        f.close()
        content = data_url[90:-45].split(';')[1]
        
        image_encoded = content.split(',')[1]
       
        out = ""
        
        body = base64.decodebytes(image_encoded.encode("utf-8"))
        fh = open("imageToSave.png", "wb")
        fh.write(body)
        fh.close()
        
        img1 = cv2.imread("imageToSave.png",cv2.IMREAD_UNCHANGED)

        img = crop(img1)
        img = img/255.0

        with graph.as_default():
            try:
                prediction = model.predict(np.expand_dims(img,axis=0))
                global out1
                out1 =  testmodel.decode_label(prediction)
                
            except Exception as e:
                
                print(e)
       
        
        ensure_pages_directory_exists()
        pages = os.listdir('pages')
        print("offline:",(out1) )
        out2 = autocorrect.spell(out1)
        #flash(out1)
        out1 = out1 + "-->" + out2
        cv2.imwrite("./images-file/" + str(time.time()) +"_"+words[:imdexn]+"_"+ ".png",img1)
        return {"result":out1}
        #flash("image saved"+str(body.shape))


@app.route('/onlrec', methods=['GET', 'POST'])
def your_view2():
    if request.method == 'POST':
        data_url = request.get_data('onlrec')   # here parse the data_url out http://xxxxx/?image={dataURL}
        
        #print(data_url)
        data_url = json.loads(data_url.decode("utf-8"))
        #print(data_url["requests"][0]["ink"])
        output = restapi.hwrrecog(data_url["requests"][0]["ink"])

        print( "vision online: ",output)
        
        return {"out":output}
        #flash("image saved"+str(body.shape))"""


app.secret_key = 'secret key'