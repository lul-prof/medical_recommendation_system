#Importing all required Libraries for the project
from flask import Flask, request , render_template, request, url_for, session, redirect, flash
from werkzeug.security import generate_password_hash, check_password_hash
from flask_mail import Mail, Message
import mysql.connector
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
import ast




#loading datasets for Disease prediction
symptoms=pd.read_csv("datasets\symtoms_df.csv")
precaution=pd.read_csv("datasets\precautions_df.csv")
workout=pd.read_csv("datasets\workout_df.csv")
description=pd.read_csv("datasets\description.csv")
medication=pd.read_csv("datasets\medications.csv")
diet=pd.read_csv("datasets\diets.csv")
diabetes=pd.read_csv("datasets\diabetes.csv")
heart_data=pd.read_csv("datasets\heart_disease_data.csv")
kidney_data=pd.read_csv("datasets\kidney Dataset.csv")

#load model for disease prediction
svc=pickle.load(open("models\svc.pkl",'rb'))

#models for diabetes and heart defect
model=pickle.load(open("models\heart.pkl",'rb'))
kidney_model=pickle.load(open("models\kidney.pkl",'rb'))
classifier=pickle.load(open("models\diabetes.sav",'rb'))




app=Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = ''
app.config['SQLALCHEMY_TRACK_MODIFICATIONS']= False
app.config['SECRET_KEY'] = ''

# SMTP Configuration (Using Gmail)
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = ''
app.config['MAIL_PASSWORD'] = ''
app.config['MAIL_DEFAULT_SENDER'] = ''

mail = Mail(app)



    

#helper function for disease prediction
def helper(dis):
    descr =  description[description['Disease']==dis]['Description']
    descr = " ".join([w for w in descr])
    
    prec=precaution[precaution['Disease']==dis][['Precaution_1','Precaution_2','Precaution_3','Precaution_4']]
    prec=[col for col in prec.values]
    
    med_string =medication[medication['Disease']==dis]['Medication'].values[0]
    med =ast.literal_eval(med_string)

    die_string=diet[diet['Disease']==dis]['Diet'].values[0]
    die=ast.literal_eval(die_string)

    work=workout[workout['disease']==dis]['workout']

    return descr,prec,med,die,work
  
   
#from our first data set i.e. training dataset
symptoms_dict = {'itching': 0, 'skin_rash': 1, 'nodal_skin_eruptions': 2, 'continuous_sneezing': 3, 'shivering': 4, 'chills': 5, 'joint_pain': 6, 'stomach_pain': 7, 'acidity': 8, 'ulcers_on_tongue': 9, 'muscle_wasting': 10, 'vomiting': 11, 'burning_micturition': 12, 'spotting_urination': 13, 'fatigue': 14, 'weight_gain': 15, 'anxiety': 16, 'cold_hands_and_feets': 17, 'mood_swings': 18, 'weight_loss': 19, 'restlessness': 20, 'lethargy': 21, 'patches_in_throat': 22, 'irregular_sugar_level': 23, 'cough': 24, 'high_fever': 25, 'sunken_eyes': 26, 'breathlessness': 27, 'sweating': 28, 'dehydration': 29, 'indigestion': 30, 'headache': 31, 'yellowish_skin': 32, 'dark_urine': 33, 'nausea': 34, 'loss_of_appetite': 35, 'pain_behind_the_eyes': 36, 'back_pain': 37, 'constipation': 38, 'abdominal_pain': 39, 'diarrhoea': 40, 'mild_fever': 41, 'yellow_urine': 42, 'yellowing_of_eyes': 43, 'acute_liver_failure': 44, 'fluid_overload': 45, 'swelling_of_stomach': 46, 'swelled_lymph_nodes': 47, 'malaise': 48, 'blurred_and_distorted_vision': 49, 'phlegm': 50, 'throat_irritation': 51, 'redness_of_eyes': 52, 'sinus_pressure': 53, 'runny_nose': 54, 'congestion': 55, 'chest_pain': 56, 'weakness_in_limbs': 57, 'fast_heart_rate': 58, 'pain_during_bowel_movements': 59, 'pain_in_anal_region': 60, 'bloody_stool': 61, 'irritation_in_anus': 62, 'neck_pain': 63, 'dizziness': 64, 'cramps': 65, 'bruising': 66, 'obesity': 67, 'swollen_legs': 68, 'swollen_blood_vessels': 69, 'puffy_face_and_eyes': 70, 'enlarged_thyroid': 71, 'brittle_nails': 72, 'swollen_extremeties': 73, 'excessive_hunger': 74, 'extra_marital_contacts': 75, 'drying_and_tingling_lips': 76, 'slurred_speech': 77, 'knee_pain': 78, 'hip_joint_pain': 79, 'muscle_weakness': 80, 'stiff_neck': 81, 'swelling_joints': 82, 'movement_stiffness': 83, 'spinning_movements': 84, 'loss_of_balance': 85, 'unsteadiness': 86, 'weakness_of_one_body_side': 87, 'loss_of_smell': 88, 'bladder_discomfort': 89, 'foul_smell_of urine': 90, 'continuous_feel_of_urine': 91, 'passage_of_gases': 92, 'internal_itching': 93, 'toxic_look_(typhos)': 94, 'depression': 95, 'irritability': 96, 'muscle_pain': 97, 'altered_sensorium': 98, 'red_spots_over_body': 99, 'belly_pain': 100, 'abnormal_menstruation': 101, 'dischromic _patches': 102, 'watering_from_eyes': 103, 'increased_appetite': 104, 'polyuria': 105, 'family_history': 106, 'mucoid_sputum': 107, 'rusty_sputum': 108, 'lack_of_concentration': 109, 'visual_disturbances': 110, 'receiving_blood_transfusion': 111, 'receiving_unsterile_injections': 112, 'coma': 113, 'stomach_bleeding': 114, 'distention_of_abdomen': 115, 'history_of_alcohol_consumption': 116, 'fluid_overload.1': 117, 'blood_in_sputum': 118, 'prominent_veins_on_calf': 119, 'palpitations': 120, 'painful_walking': 121, 'pus_filled_pimples': 122, 'blackheads': 123, 'scurring': 124, 'skin_peeling': 125, 'silver_like_dusting': 126, 'small_dents_in_nails': 127, 'inflammatory_nails': 128, 'blister': 129, 'red_sore_around_nose': 130, 'yellow_crust_ooze': 131}
diseases_list = {15: 'Fungal infection', 4: 'Allergy', 16: 'GERD', 9: 'Chronic cholestasis', 14: 'Drug Reaction', 33: 'Peptic ulcer diseae', 1: 'AIDS', 12: 'Diabetes ', 17: 'Gastroenteritis', 6: 'Bronchial Asthma', 23: 'Hypertension ', 30: 'Migraine', 7: 'Cervical spondylosis', 32: 'Paralysis (brain hemorrhage)', 28: 'Jaundice', 29: 'Malaria', 8: 'Chicken pox', 11: 'Dengue', 37: 'Typhoid', 40: 'hepatitis A', 19: 'Hepatitis B', 20: 'Hepatitis C', 21: 'Hepatitis D', 22: 'Hepatitis E', 3: 'Alcoholic hepatitis', 36: 'Tuberculosis', 10: 'Common Cold', 34: 'Pneumonia', 13: 'Dimorphic hemmorhoids(piles)', 18: 'Heart attack', 39: 'Varicose veins', 26: 'Hypothyroidism', 24: 'Hyperthyroidism', 25: 'Hypoglycemia', 31: 'Osteoarthristis', 5: 'Arthritis', 0: '(vertigo) Paroymsal  Positional Vertigo', 2: 'Acne', 38: 'Urinary tract infection', 35: 'Psoriasis', 27: 'Impetigo'}

#Disease prediction Function
def get_prediction(patient_symptoms):
    input_vector = np.zeros(len(symptoms_dict))

    for item in patient_symptoms:
        input_vector[symptoms_dict[item]]=1
        #to return actual disease instead of integer
    return diseases_list[svc.predict([input_vector])[0]] 


def format_symptom(symptom):
    return symptom.lower().replace(" ","_")



def yes_no_into_binary(value):
    #convert yes/no to 1/0 and  return 0 for empty input
    return 1 if value.lower=="yes" else 0


def sex_to_binary(value):
    #convert yes/no to 1/0 and  return 0 for empty input
    return 1 if value.lower=="male" else 0




def send_email_notification(to_email, name, diagnosis, description):
    try:
        subject = "Diagnosis Notification"
        body = f"Hello {name},\n\nYour AI Predicted Diagnosis is {diagnosis} and its description is: {description}.\n\nThank you for using our system. The diagnosis might not be 100% accurate. Visit a doctor if symptoms are serious."
        msg = Message(subject, recipients=[to_email], body=body)
        mail.send(msg)
        return "Email sent successfully!"
    except Exception as err:
        flash(f"Email Sending Error:{str(err)}","error")


def send_appointment_notification(to_email, name, day,time):
    try:
        subject = "Appointment Notification"
        body = f"Hello {name},\n\nYour Appointent has been booke on: {day} at Exactly {time}.\n\nThank you for using our system."
        msg = Message(subject, recipients=[to_email], body=body)
        mail.send(msg)
        return "Email sent successfully!"
    except Exception as err:
        flash(f"Email Sending Error:{str(err)}","error")

def send_registration_notification(to_email, fname):
    try:
        subject = "Account creation Notification"
        body = f"Hello {fname},\n\nYour Account creation was successful"
        msg = Message(subject, recipients=[to_email], body=body)
        mail.send(msg)
        return "Email sent successfully!"
    except Exception as err:
        flash(f"Email Sending Error:{str(err)}","error")


def send_login_notification(to_email, username):
    try:
        subject = "Login Notification"
        body = f"Hello {username},\n\nYour Login was successful"
        msg = Message(subject, recipients=[to_email], body=body)
        mail.send(msg)
        return "Email sent successfully!"
    except Exception as err:
        flash(f"Email Sending Error:{str(err)}","error")

def send_help_notification(to_email, names):
    try:
        subject = "HELP Notification"
        body = f"Hello {names},\n\nYour Question was received successfully. We usually answer back in less than 48 hours."
        msg = Message(subject, recipients=[to_email], body=body)
        mail.send(msg)
        return "Email sent successfully!"
    except Exception as err:
        flash(f"Email Sending Error:{str(err)}","error")


def send_helpReply_notification(to_email, names, response, question):
    try:
        subject = "HELP Notification"
        body = f"Hello {names},\n\nYour Question {question} has been answeed. {response}"
        msg = Message(subject, recipients=[to_email], body=body)
        mail.send(msg)
        return "Email sent successfully!"
    except Exception as err:
        flash(f"Email Sending Error:{str(err)}","error")

def send_doctor_notification(to_email, name,specialty,day,password):
    try:
        subject = "Employment Notification"
        body = f"Hello {name},\n\nYour employment as doctor specializing in{specialty} was successful.\n\n You are expected to report to work every{day} and your password is{password}"
        msg = Message(subject, recipients=[to_email], body=body)
        mail.send(msg)
        return "Email sent successfully!"
    except Exception as err:
        flash(f"Email Sending Error:{str(err)}","error")

# Database Connection Function
def get_db_connection():
    try:
        return mysql.connector.connect(
            host="localhost",
            user="root",
            password="",
            database="medical_records"
        )
    except Exception as e:
        flash(f"Database Error:{str(e)}","error")




#fetch kidney records from DB table for printing purposes
def get_kidney_records():
    try:
        conn=get_db_connection()
        cursor=conn.cursor(dictionary=True)
        cursor.execute("SELECT `id`,`name`, `age`, `blood_glucose`, `kidney_diagnosis`,`doctor` FROM `kidney_test`")
        records=cursor.fetchall()

    except mysql.connector.Error as err:
        flash('The Following Error Occured:'+str(err),'error')
    
    return records



#fetch diabetes records from DB table for printing purposes
def get_diabetes_records():
    try:
        conn=get_db_connection()
        cursor=conn.cursor(dictionary=True)
        cursor.execute("SELECT `id`,`name`, `age`, `insulin`, `diab_diagnosis`,`doctor` FROM `diabetes_test`")
        records=cursor.fetchall()

    except mysql.connector.Error as err:
            flash('The Following Error Occured:'+str(err),'error')


    return records


#fetch heart records from DB table for printing purposes
def get_heart_records():
    try:
        conn=get_db_connection()
        cursor=conn.cursor(dictionary=True)
        cursor.execute("SELECT `id`,`name`, `age`, `cholestral`, `heart_diagnosis`,`doctor` FROM `heart_test`")
        records=cursor.fetchall()

    except mysql.connector.Error as err:
        flash(f'The following Error occured:{str(err)},' 'error')


    return records


#fetch self Diagnosis records from DB table for printing purposes
def get_diagnosis_records():
    try:
        conn=get_db_connection()
        cursor=conn.cursor(dictionary=True)
        cursor.execute("SELECT `id`,`name`, `diagnosis`, `time`, `symptoms` FROM `self_diagnosis`")
        records=cursor.fetchall()

    except mysql.connector.Error as err:
        flash(f'The following Error occured:{str(err)},' 'error')


    return records

def get_specialty():
    try:
        conn=get_db_connection()
        cursor=conn.cursor(dictionary=True)
        cursor.execute("SELECT `id`,`name`, `description` FROM `specialty`")
        specialty=cursor.fetchall()

    except mysql.connector.Error as err:
        flash(f'The following Error occured:{str(err)},' 'error')

    return specialty



def get_doctor():
    try:
        conn=get_db_connection()
        cursor=conn.cursor(dictionary=True)
        cursor.execute("SELECT `id`,`full_names`, `specialty`,`contact`,`day`,`password`  FROM `register`")
        doctors=cursor.fetchall()

    except mysql.connector.Error as err:
        flash(f'The following Error occured:{str(err)},' 'error')


    return doctors



def get_admin():
    try:
        conn=get_db_connection()
        cursor=conn.cursor(dictionary=True)
        cursor.execute("SELECT `id`,`username`, `password1`, `Time_created` FROM `admin`")
        admins=cursor.fetchall()

    except mysql.connector.Error as err:
        flash(f'The following Error occured:{str(err)},' 'error')


    return admins

def get_patient():
    try:
        conn=get_db_connection()
        cursor=conn.cursor(dictionary=True)
        cursor.execute("SELECT `id`,`fname`, `lname`, `email`, `password` FROM `patient_register`")
        patients=cursor.fetchall()

    except mysql.connector.Error as err:
        flash(f'The following Error occured:{str(err)},' 'error')


    return patients


diabetes_remedies = [
    "Diet: Balanced diet",
    "Exercise: To help improve insulin & blood pressure control",
    "Weight Management",
    "Stress Management",
    "Medication: Metformin, Sulfonylureas, Glitazones, Glinides, GLP-1, SGLT2 inhibitors, DPP-4 inhibitors",
    "Insulin Therapy: Insulin Administration",
    "Monitoring: Continuous glucose & blood glucose monitoring"
]

kidney_remedies = [
    "Diatery changes: limit sodium, phosphorus  and  potassium intake",
    "Exercise",
    "Weight Management",
    "Medications: ACE inhibitors and ARBs, SGLT2 inhibitors",
    "Dialysis",
    "Kidney Transplant",
    "Addressing underlying condition (diabetes and high bp)"
]

heart_remedies = [
    "Diet: low in saturated trans fats, sodium and choletral",
    "Exercise: Regular Physical Exercise",
    "Smoking Cessation",
    "Weight Management",
    "Medication: ACE inhibitors and ARBs, SGLT2 inhibitors, Beta blockers, Diuretics, Statins, Antiplatelets",
    "Insulin Therapy: Insulin Administration",
    "Heart Transplant"
]




#routes For our webpages, buttons and links.
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/AIDOC')
def AIDOC():
    return render_template('AIDOC.html')

@app.route('/diabetes')
def diabetesBtn():
    return render_template('diabetes.html')

@app.route('/heart')
def heartBtn():
    return render_template('heart.html')

@app.route('/kidney')
def kidneyBtn():
    return render_template('kidney.html')

@app.route('/view_doctors')
def ViewDoc():
    doctors=get_doctor()
    return render_template('viewDoctors.html', doctors=doctors)

@app.route('/view_patients')
def ViewPat():
    patients=get_patient()
    return render_template('viewPatients.html', patients=patients)

@app.route('/view_admins')
def ViewAdm():
    admins=get_admin()
    return render_template('viewAdmins.html',admins=admins)


@app.route('/kidneyresults')
def KidResultBtn():
    if 'username' not in session:
        flash("Login First","Error")
        return redirect(url_for('Login'))
    else:
        data=get_kidney_records()
    return render_template('kidneyresults.html', records=data)


@app.route('/diabetesresults')
def DiabResultBtn():
    if 'username' not in session:
        flash("Login First","Error")
        return redirect(url_for('Login'))
    else:
        data=get_diabetes_records()
    return render_template('diabetesresults.html', records=data)



@app.route('/heartresults')
def HeartResultBtn():
    if 'username' not in session:
        flash("Login First","Error")
        return redirect(url_for('Login'))
    else:
        data=get_heart_records()
    return render_template('heartresults.html', records=data)



@app.route('/self_diag_records')
def SelfDiagBtn():
    if 'username' not in session:
        flash("Login First","Error")
        return redirect(url_for('Login'))
    else:
        data=get_diagnosis_records()
    return render_template('selfdiagnosisrecords.html', records=data)

@app.route('/admin')
def admin():
    if 'a_username' not in session:
        flash("Login To continue!!!", "success")
        return redirect(url_for('adminLogin'))
    
    return render_template("admin.html")

@app.route('/admin_login', methods=['GET','POST'])
def adminLogin():
    try:
        if request.method=='POST':
            username=request.form.get('username')
            password=request.form.get('password')

            
            conn = get_db_connection()
            cursor = conn.cursor(dictionary=True)
            cursor.execute("SELECT `id`,`username`,`password` FROM `admin` WHERE username=%s",(username,))
            admin = cursor.fetchone()
            conn.close()

            if admin:
                if password==admin['password']:
                    session['a_username']=admin['username']
                    a_username=session['a_username']
                    flash(f"Welcome {admin['username']}", "success")
                    return redirect(url_for('admin'))
                else:
                    flash("Incorrect password!!","success")
                    return redirect(url_for('adminLogin'))
            else:
                flash("No user with that Username Exists!!!")
                return redirect(url_for('admin'))

    except Exception as err:
        flash(f"Check:{err}","success")
        return redirect(url_for('admin'))
    
    return render_template('admin.html')


@app.route('/add_admin', methods=['GET','POST'])
def adminAdd():
    try:
        if request.method=='POST':
            username=request.form.get('username')
            password=request.form.get('password')

            password1=generate_password_hash(password)
            conn = get_db_connection()
            cursor = conn.cursor(dictionary=True)
            username=username
            password=password
            cursor.execute("INSERT INTO `admin` (`username`,`password`,`password1`) VALUES (%s,%s,%s)",(username,password,password1))
            conn.commit()
            flash(f"{username} Added sucessfully.Password{password}. Now Login","success")
            return render_template('admin.html')
        else:
            return render_template('addAdmin.html')

    except Exception as err:
        flash(f"Check:{err}","success")
        return redirect(url_for('admin'))

@app.route('/a_self_diag_records')
def ASelfDiagBtn():
    if 'a_username' not in session:
        flash("Login First","Error")
        return redirect(url_for('admin'))
    else:
        data=get_diagnosis_records()
    return render_template('adminselfdiagnosisrecords.html', records=data)

@app.route('/a_kidneyresults')
def KidnResultBtn():
    if 'a_username' not in session:
        flash("Login First","Error")
        return redirect(url_for('admin'))
    else:
        data=get_kidney_records()
    return render_template('adminkidneyresults.html', records=data)


@app.route('/a_diabetesresults')
def DiabeResultBtn():
    if 'a_username' not in session:
        flash("Login First","Error")
        return redirect(url_for('admin'))
    else:
        data=get_diabetes_records()
    return render_template('admindiabetesresults.html', records=data)



@app.route('/a_heartresults')
def HeartAResultBtn():
    if 'a_username' not in session:
        flash("Login First","Error")
        return redirect(url_for('admin'))
    else:
        data=get_heart_records()
    return render_template('adminheartresults.html', records=data)


@app.route('/a_logout')
def adminLogout():
    a_username=session['a_username']
    session.pop('a_username', None)
    flash(a_username+"--Logged Out Successfully","success")

    return redirect(url_for('index'))

@app.route('/response', methods=['GET','POST'])
def response():
    try:
        conn=get_db_connection()
        cursor=conn.cursor(dictionary=True)
        cursor.execute("SELECT `id`,`names`, `email`,`contact`, `question` FROM `questions`")
        questions=cursor.fetchall()

        if request.method=='POST':
            user=request.form.get('user')
            email=request.form.get('email')
            question=request.form.get('question')
            response=request.form.get('response')

            user=user
            question=question
            response=response
            email=email

            cursor.execute("INSERT INTO `responses` (`user`,`question`, `response`) VALUES (%s,%s, %s)",(user, question, response))
            conn.commit()

            send_helpReply_notification(email, user, question, response)
            if send_help_notification(email, user):
                flash("Help Message send to Clients email successfully","success")
            else:
                flash("Error Sending Email. Check:","error")

            flash(f"Response Updated Successfully","success")

        else:
            flash("Respond to FAQS", "success")

    except Exception as err:
        flash(f"Check {err}", "success")


    return render_template('response.html', questions=questions)

@app.route('/appointments', methods=['GET','POST'])
def appoitments():
    try:
        conn=get_db_connection()
        cursor=conn.cursor(dictionary=True)
        cursor.execute("SELECT `id`,`name`,`email`,`day`,`specialty` FROM `appointment`")
        patients=cursor.fetchall()

        cursor.execute("SELECT `id`,`full_names`, `specialty`,`day`,`time_in`,`time_out` FROM `register`")
        doctors=cursor.fetchall()

        if request.method=='POST':
            patient=request.form.get('patient')
            email=request.form.get('email')
            day=request.form.get('day')
            time=request.form.get('time')
            specialist=request.form.get('specialist')

            patient=patient
            email=email
            day=day
            time=time
            specialist=specialist
            name=patient

            cursor.execute("INSERT INTO `booked_appointments` (`patient`,`email`,`specialist`, `day`, `time`) VALUES (%s,%s, %s, %s, %s)",(patient, email,specialist,day,time))
            conn.commit()
           
            send_appointment_notification(email,name,day,time)
            if send_appointment_notification(email,name,day,time):
                flash("Message send to Patient's email successfully","success")
            else:
                flash("Error Sending Email. Check:","error")
            

            flash("Appointment Booked Successfuly!","success")
            
           


        else:
            flash("Respond to Appointments", "success")

    except Exception as err:
        flash(f"Check {err}", "success")


    return render_template('appointment.html', patients=patients, doctors=doctors)

@app.route('/bookings')
def Booked():
    return render_template('doc_appointments.html')

@app.route('/doc_appointment')
def DocAppointments():
    if 'username' not in session:
        flash('Login To Continue!!!')
        return redirect(url_for('Login'))
    
    try:
        username=session['username']
        conn=get_db_connection()
        cursor=conn.cursor(dictionary=True)
        cursor.execute("SELECT `id`,`full_names`, `specialty`, `username` FROM `register` WHERE `username` =%s",(username,))
        doctor=cursor.fetchone()
        if not doctor:
            flash("Doctor not found!", "danger")
            return redirect(url_for('Login'))
        
        specialty=doctor['specialty']
        cursor.execute("SELECT `id`,`patient`,`specialist`,`day`,`time` FROM `booked_appointments` WHERE `specialist` = %s", (specialty,))
        appointments=cursor.fetchall()

        if not appointments:
            flash("No appointments booked yet.", "success")
        else:
            flash('Appointment Fetched Successfully',"success")
            return render_template('doc_appointments.html', appointments=appointments,username=username)
                
    except Exception as err:
        flash(f"Check {err}","success")

    return render_template('doc_appointments.html', appointments=appointments,username=username)


@app.route('/patient_appointment')
def PatientAppointments():
    if 'p_username' not in session:
        flash('Login Or Register  to access more features!')
       
      
    try:
        username=session['p_username']
        conn=get_db_connection()
        cursor=conn.cursor(dictionary=True)
        cursor.execute("SELECT `id`,`fname`,`email`,`username` FROM `patient_register` WHERE `username` =%s",(username,))
        patient=cursor.fetchone()
        if not patient:
            flash("Kindly register to access more features!", "error")
        
        email=patient['email']
        cursor.execute("SELECT `id`,`patient`,`email`,`specialist`,`day`,`time` FROM `booked_appointments` WHERE `email` = %s", (email,))
        appointment=cursor.fetchall()

        if not appointment:
            flash("No appointments booked yet.", "success")
        else:
            flash(f'Appointment Fetched Successfully {username}',"success")
            return render_template('contacts.html', appointment=appointment,username=username)
                
    except Exception as err:
        flash(f"Check {err}","success")

    return render_template('contacts.html')

@app.route('/responses')
def responses():
    responses=""
    try:
        conn=get_db_connection()
        cursor=conn.cursor(dictionary=True)
        cursor.execute("SELECT `id`,`user`, `question`, `response` FROM `responses`")
        responses=cursor.fetchall()

    except Exception as err:
        flash(f"Check {err}", "success")
    
    return render_template('responses.html', responses=responses)

@app.route('/view_records/<int:id>')
def view_records(id):
    if 'username' not in session:
        flash("Login First","Error")
        return redirect(url_for('Login'))
    else:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT * FROM self_diagnosis WHERE id = %s", (id,))
        record = cursor.fetchone()
        conn.close()
    return render_template('view-selfD.html', record=record)
    

@app.route('/edit_records/<int:id>', methods=['GET', 'POST'])
def edit_records(id):
    if 'username' not in session:
        flash("Login First","Error")
        return redirect(url_for('Login'))
    else:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT * FROM self_diagnosis WHERE id = %s", (id,))
        record = cursor.fetchone()
        conn.close()
    
        if request.method == 'POST':
            try:
                name = request.form['name']
                email = request.form['email']
                diagnosis = request.form['diagnosis']
                description = request.form['description']
            
                conn = get_db_connection()
                cursor = conn.cursor()
                cursor.execute("UPDATE self_diagnosis SET name=%s, email=%s, diagnosis=%s, description=%s, WHERE id=%s",
                            (name, email, diagnosis, description, id))
                conn.commit()
                conn.close()
                flash("Records updated successfully!", "success")
                return redirect(url_for('SelfDiagBtn'))
            except Exception as e:
                flash(f"Error: {str(e)}", "error")
    
    return render_template('update.html', record=record)

@app.route('/delete_records/<int:id>')
def delete_records(id):
    if 'username' not in session:
        flash("Login First","Error")
        return redirect(url_for('Login'))
    else:
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute("DELETE FROM self_diagnosis WHERE id = %s", (id,))
            conn.commit()
            conn.close()
            flash("Record deleted successfully!", "success")
        except Exception as e:
            flash(f"Error: {str(e)}", "error")

    return redirect(url_for('SelfDiagBtn'))


@app.route('/diabetesTest', methods=['POST','GET'])
def TestDiabetes():
     diab_prediction=''
     diab_diagnosis=''
     if 'username' not in session:
        flash("Please log in first", "error")
        return redirect(url_for('nurse'))
     
     doctor=session['username']
     if request.method=='POST':
        try:
            name=request.form.get('name')
            user_input=[
                float(request.form.get('pregnancy')),
                float(request.form.get('gluc')),
                float(request.form.get('bp')),
                float(request.form.get('skt')),
                float(request.form.get('insulin')),
                float(request.form.get('bmi')),
                float(request.form.get('pedigree')),
                float(request.form.get('age'))
            ]
            diab_diagnosis=''
            diab_prediction=classifier.predict([user_input])

            if diab_prediction[0] == 0:
               diab_diagnosis='The Person is Not Diabetic'
            
            else:
                diab_diagnosis='The person is diabetic'
               

            #Conection To the Database!!!!!!
            conn = get_db_connection()
                        
            cursor = conn.cursor(dictionary=True)

            name=name
            insulin = user_input[4]
            age = user_input[7]
            diab_diagnosis=diab_diagnosis
            doctor=doctor

            cursor.execute("INSERT INTO `diabetes_test` (`name`,`age`, `insulin`, `diab_diagnosis`, `doctor`) VALUES (%s,%s, %s, %s, %s)",(name, age, insulin, diab_diagnosis, doctor))
            conn.commit()

            flash("Test Result saved Successfuly!","success")

            #diabetes_remedies=(Diet: balanced diet, Excercise: to help improve insulin&bp control, Weight management, Stress management, Medication: Metformin Sulfonylureas Glitazones Glinides GLP-1 SGLT2 inhibitors DPP-4 inhibitors,Insulin Therapy: Insuling Administartion, Continous glucose&blood glucose monitoring)


        except Exception as err2:
            flash(f'Check:{str(err2)}','error')

     return render_template('diabetes.html',diab_diagnosis=diab_diagnosis, remedies=diabetes_remedies) 



@app.route('/HeartTest', methods=['POST','GET'])
def TestHeart():
    if 'username' not in session:
        flash("Please log in first", "error")
        return redirect(url_for('nurse'))
    
    doctor=session['username']
    try:
        if request.method=='POST':
                name=request.form.get('name')
                age=request.form.get('age')
                sex=sex_to_binary(request.form.get('sex',"male"))
                cp=request.form.get('cp')
                trestbps=request.form.get('rbp')
                chol=request.form.get('chol')
                fbs=request.form.get('fbs')
                restecg=request.form.get('rer')
                thalach=request.form.get('mhr')
                exang=request.form.get('eia')
                oldpeak=request.form.get('st')
                slope=request.form.get('slope')
                ca=request.form.get('vessels')
                thal=request.form.get('defects')

                #code for prediction
                heart_diagnosis=''

                user_input = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]
                
                user_input = [float(x) for x in user_input]
            
                heart_prediction = model.predict([user_input])


                if heart_prediction[0] == 1:
                    heart_diagnosis = 'The person is having heart disease'
                  
                else:
                    heart_diagnosis = 'The person does not have any heart disease'
                #Conection To the Database!!!!!!
                conn = get_db_connection()     
                cursor = conn.cursor(dictionary=True)

                name=name
                age = age
                sex=sex
                cholestral = chol
                heart_diagnosis=heart_diagnosis
                doctor=doctor

                cursor.execute("INSERT INTO `heart_test` (`name`,`age`, `cholestral`, `heart_diagnosis`, `doctor`) VALUES (%s,%s,%s,%s,%s)",(name, age, cholestral, heart_diagnosis,doctor))

                conn.commit()
                print(f"sex:",sex)
                flash("Test Result saved Successfuly!","success")

                #heart_remedies=[Diet: low in saturated trans fats, sodium&choletral,Exercise: Regular Physical Exercise, Smoking Cessation, Weight Management, Stress Management,Medication: ACE inhibitors&ARBs SGLT2 inhibitors Beta blockers Diuretics Statins Antiplatelets, Heart Transplant]


    except Exception as err2:
        flash(f'Check:{str(err2)}','error')


    return render_template('heart.html',heart_diagnosis=heart_diagnosis, remedies=heart_remedies)




@app.route('/KidneyTest', methods=['POST','GET'])
def TestKidney():
    #Declare Variables
    kidney_diagnosis=''
    kidney_prediction=''
    if 'username' not in session:
        flash("Please log in first", "error")
        return redirect(url_for('nurse'))
    
    doctor=session['username']

    if request.method=='POST':
        try:
            name=request.form.get('name')
            user_input=[
                float(request.form.get('age') or 0),
                float(request.form.get('blood_pressure') or 0),
                float(request.form.get('specific_gravity') or 0),
                float(request.form.get('albumin') or 0),
                float(request.form.get('sugar') or 0),
                float(request.form.get('blood_glucose') or 0),
                float(request.form.get('blood_urea')),
                float(request.form.get('serum_creatinine') or 0),
                float(request.form.get('sodium') or 0),
                float(request.form.get('potassium') or 0),
                float(request.form.get('hemoglobin')),
                float(request.form.get('packed_cell_volume') or 0),
                float(request.form.get('white_bc') or 0),
                float(request.form.get('red_bc') or 0),
                float(request.form.get('rbc') or 0),
                float(yes_no_into_binary(request.form.get('pus_cells_normal',"no")) or 0),
                float(yes_no_into_binary(request.form.get('puss_cell_clumps_present',"no")) or 0),
                float(yes_no_into_binary(request.form.get('Bacteria_present',"no")) or 0),
                float(yes_no_into_binary(request.form.get('hypertension',"no")) or 0),
                float(yes_no_into_binary(request.form.get('diabetes_mellitus',"no")) or 0),
                float(yes_no_into_binary(request.form.get('coronary_artery_disease',"no")) or 0),
                float(yes_no_into_binary(request.form.get('appetite',"no")) or 0),
                float(yes_no_into_binary(request.form.get('radal_edema',"no")) or 0),
                float(yes_no_into_binary(request.form.get('anaemia',"no")) or 0)
            ]

            kidney_prediction=kidney_model.predict([user_input])

            if kidney_prediction[0] == 0:
               kidney_diagnosis='The Person does not have kidney issues'
            
            else:
                kidney_diagnosis='The person has kidney issues'
               
            #Conection To the Database!!!!!!
            conn = get_db_connection()   
            cursor = conn.cursor(dictionary=True)

            name=name
            age = user_input[0]
            blood_glucose = user_input[5]
            kidney_diagnosis=kidney_diagnosis
            doctor=doctor

            cursor.execute("INSERT INTO `kidney_test` (`name`,`age`, `blood_glucose`, `kidney_diagnosis`, `doctor`) VALUES (%s,%s, %s, %s, %s)",(name, age, blood_glucose, kidney_diagnosis,doctor))

            conn.commit()

            flash("Test Result saved Successfuly!","success")

            #kidney_remedies=[Diatery changes: limit sodium phosphorus&potassium intake, Exercise ,Weight Management, Medications: ACE inhibitors&ARBs&SGLT2 inhibitors, Dialysis,Kidney Transplant,Addressing underlying condition like diabetes&high bp]


        except Exception as err:
            flash(f'Check:{str(err)}','error')

    
    return render_template('kidney.html',kidney_diagnosis=kidney_diagnosis, remedies=kidney_remedies) 


@app.route('/predict', methods=['POST','GET']) 
def predict():
    p_username=session['p_username']
    predicted_disease=''
    descr=''
    my_prec=''
    med=''
    die=''
    work=''
    if 'p_username' not in session:
        flash("Login To proceed")
        return redirect(url_for('PatientLogin'))
    
    if request.method=='POST':
            try:
                name=request.form.get('name')
                email=request.form.get('email')
                symptoms=request.form.get('symptoms')
                user_symptoms=[format_symptom(s.strip()) for s in symptoms.split(',')]
                user_symptoms=[sym.strip("[]' ") for sym in user_symptoms]
                predicted_disease=get_prediction(user_symptoms)
                descr,prec,med,die,work=helper(predicted_disease)

                my_prec= []
                for i in prec[0]:
                    my_prec.append(i)


                if predicted_disease:
                    flash("Diagnosis was successful")

                    conn = get_db_connection()     
                    cursor = conn.cursor(dictionary=True)

                    name=name
                    email=email
                    description=descr
                    symptoms=symptoms
                    diagnosis=predicted_disease

                    cursor.execute("INSERT INTO `self_diagnosis` (`name`,`email`, `diagnosis`,`description`,`symptoms`) VALUES (%s,%s,%s,%s,%s)",(name,email,diagnosis,description,symptoms))
                            
                    session['email']=email
                    p_username=session['p_username']

                    conn.commit()

                    flash("Test Result saved Successfuly!","success")

                    send_email_notification(email, name, diagnosis, description)
                    if send_email_notification(email, name, diagnosis, description):
                        flash("Message send to your email successfully","success")
                    else:
                        flash("Error Sending Email. Check:","error")
                        
                    
                else:
                    flash("An Error Occured while Trying to make a Doagnosis!!!","Success")
            
            except Exception as err:
                flash(f'Check: {str(err)}','error')
                     
    return render_template('AIDOC.html', predicted_disease=predicted_disease,dis_descr=descr,dis_prec=my_prec,dis_med=med,dis_diet=die, dis_work=work, p_username=p_username)


@app.route('/mine')
def Mine():
    diag=''
    try:
        email=session['email']
        #To get and print specific user Medical History.
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT id, name, diagnosis, description, symptoms FROM self_diagnosis WHERE email=%s ORDER BY id DESC LIMIT 1", (email,))
        diag = cursor.fetchone()

    except Exception as err:
        flash(f'Error: {str(err)}', 'error')


    return render_template('diagPrint.html', diag=diag)


@app.route('/contacts')
def contacts():
    return render_template('contacts.html')




@app.route('/register', methods=['GET','POST'])
def Register():
    password=''
    if request.method=='POST':
        try:
            fname = request.form.get('fname')
            lname = request.form.get('lname')
            username = request.form.get('username')
            email = request.form.get('email')
            password1 = request.form.get('password1')
            password2 = request.form.get('password2')
         
            if password1==password2:
                password=generate_password_hash(password2)
            else:
                flash("Passwords Do Not Match. Try Again!")

             #Conection To the Database!!!!!!
          
            conn = get_db_connection()  

            cursor = conn.cursor(dictionary=True)

            fname=fname
            lname=lname
            username=username
            email=email
            password=password
            password1=password1

           
               
            session['username']=username

            cursor.execute("INSERT INTO `register` (`fname`,`lname`,`username`,`email`,`password`,`password1`) VALUES (%s,%s,%s,%s,%s,%s)",(fname,lname,username,email,password,password1))

            conn.commit()

            send_registration_notification(email, fname)
            if send_registration_notification(email, fname):
                flash("Message send to your email successfully","success")
            else:
                flash("Error Sending Email. Check:","error")

            flash(username+" "+"Your Sign Up was Successful!","success")   
            
            return redirect(url_for('Login'))
   
        except Exception as err:
            flash(f"Check:{str(err)}","error")

    return render_template('signup.html')


@app.route('/login',methods=['GET','POST'])
def Login():
    if request.method=='POST':
        try:
            email=request.form.get('email')
            password=request.form.get('password')

            #DB Connection
            conn=get_db_connection()

            cursor=conn.cursor(dictionary=True)
            cursor.execute("SELECT `id`,`username`,`email`,`password`,`password1` FROM `register` WHERE email=%s",(email,))
            user=cursor.fetchone()

            if user:
                if user['password1']==password:
                    session['user_id']=user['id']
                    session['username']=user['username']
                    session['email']=user['email']
                    session['password']=user['password']

                    username=session['username']
                    d_email=session['email']
                    d_password=session['password']


                    cursor.execute("INSERT INTO `login` (`username`,`email`,`password`) VALUES (%s,%s,%s)",(username, d_email, d_password))
                    conn.commit()

                    send_login_notification(email, username)
                    if send_login_notification(email, username):
                        flash(f"Login Message send to your email successfully {username}","success")
                    else:
                        flash("Error Sending Email. Check:","error")

                    flash(f"Welcome{username}","success")
                    return render_template('Doc.html', username=username)
                
                else:
                    flash("Incorrect Password","error")

            else:
                flash("No user with that email exists","error")
            
                
        except Exception as err:
            flash(f"Check:{str(err)}","error")
            
    return render_template('login.html')

@app.route('/patient_login',methods=['GET','POST'])
def PatientLogin():
    if request.method=='POST':
        try:
            email=request.form.get('email')
            password=request.form.get('password')

            #DB Connection
            conn=get_db_connection()

            cursor=conn.cursor(dictionary=True)
            cursor.execute("SELECT `id`,`username`,`email`,`password`,`password1` FROM `patient_register` WHERE email=%s",(email,))
            user=cursor.fetchone()

            if user:
                if user['password1']==password:
                    session['p_user_id']=user['id']
                    session['p_username']=user['username']
                    session['p_password']=user['password']
                    session['p_email']=user['email']

                    p_username=session['p_username']
                    p_password=session['p_password']
                    p_email=session['p_email']


                    cursor.execute("INSERT INTO `login` (`username`,`email`,`password`) VALUES (%s,%s,%s)",(p_username, p_email, p_password))
                    conn.commit()

                    send_login_notification(email, p_username)
                    if send_login_notification(email,p_username):
                        flash(f"Login Message send to your email successfully {p_username}","success")
                    else:
                        flash("Error Sending Email. Check:","error")
                    
                    return redirect(url_for('predict'))
                
                else:
                    flash("Incorrect Password","error")

            else:
                flash("No user with that email exists","error")
            
                
        except Exception as err:
            flash(f"Check:{str(err)}","error")
            
    return render_template('AIDOC.html')

@app.route('/patient_register', methods=['GET','POST'])
def PatientRegister():
    if request.method=='POST':
        try:
            fname = request.form.get('fname')
            lname = request.form.get('lname')
            username = request.form.get('username')
            email = request.form.get('email')
            contact = request.form.get('contact')
            password1 = request.form.get('password1')
            password2 = request.form.get('password2')
         
            if password1==password2:
                password=generate_password_hash(password2)
            else:
                flash("Passwords Do Not Match. Try Again!")

             #Conection To the Database!!!!!!
          
            conn = get_db_connection()  

            cursor = conn.cursor(dictionary=True)

            fname=fname
            lname=lname
            username=username
            email=email
            contact=contact
            password=password
            password1=password1
               

            cursor.execute("INSERT INTO `patient_register` (`fname`,`lname`,`username`,`email`,`password`,`contact`,`password1`) VALUES (%s,%s,%s,%s,%s,%s,%s)",(fname,lname,username,email,password,contact,password1))

            conn.commit()

            send_registration_notification(email, fname)
            if send_registration_notification(email, fname):
                flash("Message send to your email successfully","success")
            else:
                flash("Error Sending Email. Check:","error")

            flash(f"{username} Your Sign Up was Successful!","success")   
            
            return redirect(url_for('PatientLogin'))
   
        except Exception as err:
            flash(f"Check:{str(err)}","error")

    return render_template('register.html')


@app.route('/logout')
def Logout():
    username=session['username']
    session.pop('username', None)
    flash(username+"--Logged Out Successfully","success")

    return redirect(url_for('index'))

@app.route('/p_logout')
def PatientLogout():
    p_username=session['p_username']
    session.pop('p_username', None)
    flash(p_username+"--Logged Out Successfully","success")

    return redirect(url_for('index'))

@app.route('/nurse')
def nurse():
    if 'username' not in session:
        flash("Login to cotinue!!!","success")
        return redirect(url_for('Login'))
    
    return render_template('Doc.html')


@app.route('/doctors', methods=['GET','POST'])
def Doctors():
    data=get_specialty()
    if 'a_username' not in session:
        flash("Please log in first", "error")
        return redirect(url_for('admin'))
    else:
        a_username=session['a_username']
        flash("Welcome:"+" "+a_username)

    if request.method=='POST':
        try:
            name=request.form.get('name')
            age=request.form.get('age')
            specialty=request.form.get('specialty')
            email=request.form.get('email')
            username=request.form.get('username')
            idno=request.form.get('idno')
            contact=request.form.get('contact')
            day=request.form.get('day')
            time_in=request.form.get('from')
            time_out=request.form.get('to')
            password1=request.form.get('password')

            password=generate_password_hash(password1)
            #Conection To the Database!!!!!
            conn=get_db_connection()             
            cursor = conn.cursor(dictionary=True)
            name=name
            age = age
            specialty=specialty
            idno=idno
            email=email
            username=username
            contact=contact
            password1=password1
            password=password
            day=day
            time_in=time_in
            time_out=time_out

        
            cursor.execute("INSERT INTO `register` (`full_names`,`age`, `specialty`,`contact`,`day`,`time_in`,`time_out`,`national_id`,`username`,`email`,`password`,`password1`) VALUES (%s,%s, %s, %s,%s,%s,%s,%s,%s,%s,%s,%s)",(name,age,specialty,contact,day,time_in,time_out,idno,username,email,password,password1))
    
            conn.commit()

            send_doctor_notification(email, name, specialty, day, password1)
            if send_doctor_notification(email, name, specialty, day, password1):
                flash("Message send to your email successfully","success")
            else:
                flash("Error Sending Email. Check:","error")

            flash("Doctor saved Successfuly!","success")

            
        except Exception as err:
            flash(f'Check:{str(err)}', 'error')
                

        
    return render_template('doctors.html', specialty=data)


@app.route('/appointment', methods=['GET','POST'])
def Appointment():
    if request.method=='POST':
        try:
            name=request.form.get('name')
            contact=request.form.get('contact')
            email=request.form.get('email')
            specialty=request.form.get('specialty')
            day=request.form.get('day')
            

            name=name
            email=email
            contact=contact
            specialty=specialty
            day=day

            conn=get_db_connection()
            cursor=conn.cursor(dictionary=True)        
            cursor.execute("INSERT INTO `appointment` (`name`,`email`,`contact`,`specialty`,`day`) VALUES (%s,%s, %s, %s,%s)",(name,email,contact,specialty,day))
            
            conn.commit()
            flash("Appointment Queued for Verification successfully. Check your email after 2hrs","success")


        except Exception as err:
            flash(f"Check:{str(err)}","error")

            
    return render_template('contacts.html')


@app.route('/locations')
def locations():
    return render_template('locations.html')


@app.route('/developer')
def developer():
    return render_template('developer.html')

@app.route('/help')
def help():
    return render_template('help.html')

@app.route('/problem', methods=['GET','POST'])
def problem():
    try:
        if request.method=='POST':
            names=request.form.get('name')
            email=request.form.get('email')
            contact=request.form.get('contact')
            question=request.form.get('problem')

            conn=get_db_connection()
            cursor=conn.cursor(dictionary=True)
            names=names
            email=email
            contact=contact
            question=question

            cursor.execute("INSERT INTO `questions` (`names`,`email`, `contact`,`question`) VALUES (%s,%s, %s, %s)",(names,email,contact,question))
            conn.commit()

            flash("Question Received. We will respond soon!!!","success")

            send_help_notification(email, names)
            if send_help_notification(email, names):
                flash("Help Message send to your email successfully","success")
            else:
                flash("Error Sending Email. Check:","error")

    except Exception as e:
        flash(f"Check:{str(e)}","error")


    return render_template('help.html')    
 
#main
if __name__ =="__main__":
    app.run(debug=True)



