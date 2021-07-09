from flask import Flask, render_template, request
from 可视化attention import show_self_collected_image
import os
import shutil

app = Flask(__name__)


def delete_file():
    folder_path = "./static"
    del_file = os.listdir(folder_path)
    for file in del_file:
        file_path = os.path.join(folder_path,file)
        if os.path.isfile(file_path):
            os.remove(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)



@app.route('/')
def hello():
    #删除缓存文件
    delete_file()
    return render_template('index.html')

@app.route('/run',methods=['GET','POST'])
def run():
    image = request.form['image']
    model_name = request.form["Model"]
    caption=show_self_collected_image("./自选图片/{}".format(image),model_name)
    att_path="./static/~attention_image~{}".format(image)
    original_image = "./static/~original_image~{}".format(image)

    return render_template("index.html",img =original_image,result=caption,attention=att_path)


app.secret_key = 'some key that you will never guess'

if __name__ == "__main__":
    app.run('127.0.0.1', 5000, debug = True)
