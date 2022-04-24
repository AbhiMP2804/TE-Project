from cProfile import label
from flask_wtf import FlaskForm,Form
from wtforms import  StringField,PasswordField,SubmitField,TextAreaField,RadioField,IntegerField
from wtforms.validators import Length,EqualTo,DataRequired,Email,ValidationError
import email

class RegistrationForm(FlaskForm):
    username = StringField(label="Username",validators=[DataRequired(),Length(min=3,max=25)])
    email =  StringField(label="Email",validators=[DataRequired(),Email()])
    password = PasswordField(label="Password",validators=[DataRequired(),Length(min=6,max=16)])
    confirm_password = PasswordField(label="Confirm Password",validators=[DataRequired(),EqualTo("password")])
    submit = SubmitField(label="Sign Up")
    
class LoginnForm(FlaskForm):
    email =  StringField(label="Email",validators=[DataRequired(),Email()])
    password = PasswordField(label="Password",validators=[DataRequired(),Length(min=6,max=16)])
    submit = SubmitField(label="Login")
    

class ProfileForm(Form):
    name = StringField(label="NAME",validators=[DataRequired()])
    age = IntegerField(label="AGE",validators=[DataRequired()])
    bloodgroup = StringField(label="BLOOD GROUP",validators=[DataRequired()])
    phoneNumber = IntegerField(label="CONTACT NUMBER",validators=[DataRequired()])
    gender = RadioField(label="GENDER",choices=["MALE","FEMALE"])
    address = TextAreaField(label="ADDRESS")
    prevhist = StringField(label="MEDICAL HISTORY")  
    update = SubmitField(label="SHARE AS PDF")
    savepdf = SubmitField(label="GET PDF")
