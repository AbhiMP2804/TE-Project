
from email.mime import image
from fileinput import filename
from tkinter import CENTER
from turtle import title
from wsgiref.util import request_uri
from flask import Flask, Response, redirect, render_template, session, url_for,flash,request,json,make_response
from flask_bcrypt import Bcrypt,bcrypt
from flask_login import LoginManager,UserMixin,login_required,login_user,logout_user,current_user
from sklearn.datasets import load_sample_images
from forms import RegistrationForm,LoginnForm,ProfileForm
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import PrimaryKeyConstraint, null, true
from datetime import datetime
from email.policy import default
import warnings
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os
import pickle
import pdfkit


app = Flask(__name__)
app.config["SECRET_KEY"]="thisisbraintumorwebsite"
app.config["SQLALCHEMY_DATABASE_URI"]="sqlite:///database/bt.db"
app.config['SQLALCHEMY_TRACK_MODIFICATION'] = False
app.config['UPLOAD_FOLDER'] = "static/users/"
app.config['UNSEEN_FOLDER'] = "static/users/Unseen"

loaded_model = pickle.load(open("BT-model","rb"))

wkhtmltopdf_path = r'C:\Program Files\wkhtmltopdf\bin\wkhtmltopdf.exe'
config = pdfkit.configuration(wkhtmltopdf=wkhtmltopdf_path)

db=SQLAlchemy(app)
bcrypt = Bcrypt(app)
login_manager = LoginManager(app)
login_manager.init_app(app)
login_manager.login_view = "login"

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

@login_manager.unauthorized_handler
def unauthorized():
    return redirect(url_for("register"))

class User(db.Model,UserMixin):
    id=db.Column(db.Integer,primary_key=True)
    username=db.Column(db.String(40),unique=True,nullable=False)
    email=db.Column(db.String(120),unique=True,nullable=False)
    image_file=db.Column(db.String(20),nullable=False,default="default.jpg")
    password=db.Column(db.String(60),nullable=False)
    date_created = db.Column(db.DateTime,default=datetime.utcnow)
    # folder = db.Column(db.String(80), unique=True, nullable=False)


    def __repr__(self) -> str:
        return f'{self.username} : {self.email} : {self.date_created.strftime("%d/%n/%Y, %H:%M:%S")}'
        

@app.route("/")
@app.route("/home")
def homepage():
    return render_template("index1.html")


@app.route("/register",methods=["POST","GET"])
def register():
    if current_user.is_authenticated:
        return redirect(url_for("profile"))
    form = RegistrationForm()

    if form.validate_on_submit():
        encrypted_password=bcrypt.generate_password_hash(form.password.data).decode("utf-8")
        user=User(username=form.username.data,email=form.email.data,password=encrypted_password)
        db.session.add(user)
        db.session.commit()
        def folder():
            username = request.form.get("username")
            username = username.strip().capitalize()
            user_folder = os.path.join(app.config['UPLOAD_FOLDER'], username)
            os.makedirs(user_folder)
        folder()

        flash(f"Account created Successfully for {form.username.data}",category="success")
        return redirect(url_for("login"))
    return render_template("register.html",title="Register",form=form)

@app.route("/login",methods=["POST","GET"])
def login():
    if current_user.is_authenticated:
        return redirect(url_for("profile"))
    form = LoginnForm()
    if form.validate_on_submit():
        user=User.query.filter_by(email=form.email.data).first()
        if user and bcrypt.check_password_hash(user.password,form.password.data):
            login_user(user)
            # flash(f"Login Successfully for {form.email.data}",category="success")
            return redirect(url_for("homepage"))
        else:
            flash(f"Login UnSuccessfully for {form.email.data}",category="danger")
    return render_template("login.html",title="Login",form=form)


@app.route("/profile",methods=["POST","GET"])
@login_required
def profile():
    form=ProfileForm()
    # session["name"] = request.form["name"]
    if request.method=="POST":
        if form.validate()=="False":
            flash("Please Fill out this field")
            return render_template("profile.html",form=form)
        else:
            render = render_template("updatesuccessful.html",form=form)
            pdf = pdfkit.from_string(render,False,configuration=config)
            response = make_response(pdf)
            response.headers['content-Type']='application/pdf'
            response.headers['content-Disposition']='attached; filename=Details.pdf'
            return response

    elif request.method=="GET":
        return render_template("profile.html",form=form)




@app.route("/about")
@login_required
def about():
    return render_template("aboutus.html")

@app.route("/logout")
def logout():
    logout_user()
    return redirect(url_for("homepage"))


@app.route("/test",methods=["POST","GET"])
def test():
    dec = {0: 'glioma_tumor', 1: 'meningioma_tumor',
       2: 'no_tumor', 3: 'pituitary_tumor'}
    c = 1
    if request.method == "POST":
        imagefile=request.files["imagefile"]
        if current_user.is_authenticated:
            username = current_user.username
            username = username.strip().capitalize()
            user_folder = os.path.join(app.config['UPLOAD_FOLDER'], username)
            img_path = os.path.join(user_folder,imagefile.filename)
            
        else:
            img_path =os.path.join(app.config['UNSEEN_FOLDER'], imagefile.filename)
        
        imagefile.save(img_path)
        img = cv2.imread(img_path,0)
        img1 = cv2.resize(img, (200, 200))
        img1 = img1.reshape(1, -1)/255
        p = loaded_model.predict(img1)
        c+=1
        gt=4
        mt=6
        nt=9
        pt=5
        classification = "%s"%(dec[p[0]])
        # if classification == "glioma_tumor":
        #     gt=gt+1
        # if classification == "meningioma_tumor":
        #     mt=mt+1
        # if classification == "no_tumor":
        #     nt=nt+1
        # if classification == "pitutary_tumor":
        #     pt=pt+1
             
        # x=[gt,mt,nt,pt]
        
        # det ={'glioma_tumor':x[0],'meningioma_tumor':x[1],'no_tumor':x[2],'pitutary-tumor':x[3]}
        # names=list(det.keys())
        # values = list(det.values())
        # plt.bar(range(len(x)), values, tick_label=names)
        # plt.show()
        return render_template("test.html",img_pt=img_path,prediction=classification)
        
    else:
        return render_template("test.html")




if __name__ == "__main__":
    app.run(debug=True)

