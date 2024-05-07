import streamlit as st
import pickle
import sklearn
import pandas as pd
import numpy as np
from PIL import Image
model = pickle.load(open('model_new1.sav', 'rb'))

st.title('IPL Player Salary Predictor')
#st.sidebar.header('Player Data')
image = Image.open('iplaml.jpg')
st.image(image, '')
st.write("  \n")



def display_dictionary_in_table(dictionary):
    for key, value in dictionary.items():
        st.write(f"{key} | {value}")

# Example dictionary
my_dict1 = {0: 'Aaron Finch', 1: 'Abdul Samad', 2: 'Abhijeet Tomar', 3: 'Abhimanyu Mithun', 4: 'Abhinav Sadarangani', 5: 'Abhishek Nayar', 6: 'Abhishek Sharma', 7: 'Abu Nechim Ahmed', 8: 'Adam Milne', 9: 'Adam Zampa', 10: 'Aditya Garhwal', 11: 'Aditya Tare', 12: 'Agnivesh Ayachi', 13: 'Aiden Blizzard', 14: 'Aiden Markram', 15: 'Ajantha Mendis', 16: 'Ajinkya Rahane', 17: 'Akash Deep', 18: 'Akash Singh', 19: 'Akhil Arvind Herwadkar', 20: 'Akila Dananjaya', 21: 'Akila Dhananjaya', 22: 'Akshar Rajesh Patel', 23: 'Akshay Karnewar', 24: 'Akshay Wakhare', 25: 'Akshdeep Nath', 26: 'Albie Morkel', 27: 'Alex Carey', 28: 'Alex Hales', 29: 'Alzarri Joseph', 30: 'Aman Khan', 31: 'Ambati Rayudu', 32: 'Amit Mishra', 33: 'Amit Paunikar', 34: 'Andre Russell', 35: 'Andrew Tye', 36: 'Aneeshwar Gautam', 37: 'Angelo Mathews', 38: 'Aniket Choudhary', 39: 'Anirudha Ashok Joshi', 40: 'Ankeet Bawane', 41: 'Ankit Nagendra Sharma', 42: 'Ankit Sharma', 43: 'Ankit Singh Rajpoot', 44: 'Ankush Bains', 45: 'Anmolpreet Singh', 46: 'Anrich Nortje', 47: 'Ansh Patel', 48: 'Anuj Rawat', 49: 'Anukul Roy', 50: 'Anunay Singh', 51: 'Anureet Singh', 52: 'Apoorv Vijay Wankhade', 53: 'Arjun Tendulkar', 54: 'Armaan Jaffer', 55: 'Arshdeep Singh', 56: 'Aryaman Vikram Birla', 57: 'Aryan Juyal', 58: 'Asela Gunarathna', 59: 'Ashish Nehra', 60: 'Ashish Reddy', 61: 'Ashok Dinda', 62: 'Ashok Sharma', 63: 'Ashton Turner', 64: 'Ashwin Hebbar', 65: 'Asif K M', 66: 'Atharva Taide', 67: 'Avesh Khan', 68: 'Axar Rajesh Patel', 69: 'Ayush Badoni', 70: 'B. Sai Sudharsan', 71: 'Baba Aparajith', 72: 'Baba Indrajith', 73: 'Baltej Dhanda', 74: 'Bandaru Ayyappa', 75: 'Barinder Singh Saran', 76: 'Barinder Singh Sran', 77: 'Basil Thampi', 78: 'Ben Cutting', 79: 'Ben Dunk', 80: 'Ben Dwarshuis', 81: 'Ben Hilfenhaus', 82: 'Ben Laughlin', 83: 'Benjamin Stokes', 84: 'Benny Howell', 85: 'Beuran Hendricks', 86: 'Bhanuka Rajapaksa', 87: 'Bhuvneshwar Kumar', 88: 'Billy Stanlake', 89: 'Bipul Sharma', 90: 'Brad Hodge', 91: 'Brad Hogg', 92: 'Brendan Taylor', 93: 'Brendon McCullum', 94: 'C Hari Nishaanth', 95: 'C.Hari Nishaanth', 96: 'C.M. Gautam', 97: 'Cameron Delport', 98: 'Carlos Brathwaite', 99: 'Chaitanya Bishnoi', 100: 'Chama Milind', 101: 'Chamika Karunaratne', 102: 'Chetan Sakariya', 103: 'Cheteshwar Pujara', 104: 'Chirag Suri', 105: 'Chris Green', 106: 'Chris Jordan', 107: 'Chris Lynn', 108: 'Chris Woakes', 109: 'Christopher Barnwell', 110: 'Christopher Gayle', 111: 'Christopher Morris', 112: 'Clinton McKay', 113: 'Colin De Grandhomme', 114: 'Colin Ingram', 115: 'Colin Munro', 116: 'Corey Anderson', 117: 'Dale Steyn', 118: 'Dan Christian', 119: 'Daniel Christian', 120: 'Daniel Sams', 121: 'Darcy Short', 122: 'Darren Bravo', 123: 'Darren Sammy', 124: 'Darshan Nalkande', 125: 'Daryl Mitchell', 126: 'David Miller', 127: 'David Warner', 128: 'David Wiese', 129: 'David Willey', 130: 'Dawid Malan', 131: 'Debabrata Das', 132: 'Deepak Chahar', 133: 'Deepak Hooda', 134: 'Deepak Punia', 135: 'Devdutt Padikkal', 136: 'Devon Conway'}
 
my_dict2= { 137: 'Dewald Brevis', 138: 'Dhawal Kulkarni', 139: 'Dhruv Jurel', 140: 'Dhruv Shorey', 141: 'Digvijay Deshmukh', 142: 'Dinesh Karthik', 143: 'Dinesh Salunkhe', 144: 'Dirk Nannes', 145: 'Dishant Yagnik', 146: 'Dominic Drakes', 147: 'Domnic Joseph Muthuswamy', 148: 'Dushmanta Chameera', 149: 'Dushmantha Chameera', 150: 'Dwaine Pretorius', 151: 'Dwayne Bravo', 152: 'Dwayne Smith', 153: 'Eklavya Dwivedi', 154: 'Eoin Morgan', 155: 'Evin Lewis', 156: 'Fabian Allen', 157: 'Faf Du Plessis', 158: 'Farhaan Behardien', 159: 'Fazalhaq Farooqi', 160: 'Fidel Edwards', 161: 'Finn Allen', 162: 'Gautam Gambhir', 163: 'George Bailey', 164: 'Glenn Maxwell', 165: 'Glenn Phillips', 166: 'Gurinder Sandhu', 167: 'Gurkeerat Singh Mann', 168: 'HS Sharath', 169: 'Hanuma Vihari', 170: 'Harbhajan Singh', 171: 'Hardik Pandya', 172: 'Hardus Viljoen', 173: 'Harpreet Brar', 174: 'Harry Gurney', 175: 'Harshal Patel', 176: 'Heinrich Klaasen', 177: 'Himmat Singh', 178: 'Hrithik Shokeen', 179: 'Imran Tahir', 180: 'Iqbal Abdullah', 181: 'Irfan Pathan', 182: 'Ishan Kishan', 183: 'Ishan Porel', 184: 'Ishank Jaggi', 185: 'Ishant Sharma', 186: 'Ishwar Chandra Pandey', 187: 'Isuru Udana', 188: 'J Suchith', 189: 'Jacob Oram', 190: 'Jacques Kallis', 191: 'Jagadeesha Suchith', 192: 'Jalaj Saxena', 193: 'James Faulkner', 194: 'James Neesham', 195: 'Jaskaran Singh', 196: 'Jason Behrendorff', 197: 'Jason Holder', 198: 'Jason Roy', 199: 'Jasprit Bumrah', 200: 'Jatin Saxena', 201: 'Javon Searless', 202: 'Jayant Yadav', 203: 'Jaydev Shah', 204: 'Jaydev Unadkat', 205: 'Jean-Paul Duminy', 206: 'Jeevan Mendis', 207: 'Jesse Ryder', 208: 'Jhye Richardson', 209: 'Jitesh Sharma', 210: 'Joe Denly', 211: 'Joel Paris', 212: 'Jofra Archer', 213: 'Johan Botha', 214: 'John Hastings', 215: 'Jonny Bairstow', 216: 'Jos Buttler', 217: 'Josh Hazlewood', 218: 'Joshua Philippe', 219: 'Juan Theron', 220: 'K.Bhagath Varma', 221: 'K.C Cariappa', 222: 'K.C. Cariappa', 223: 'K.K. Jiyaz', 224: 'K.M. Asif', 225: 'K.S. Bharat', 226: 'KL Rahul', 227: 'Kagiso Rabada', 228: 'Kamlesh Nagarkoti', 229: 'Kane Richardson', 230: 'Kane Williamson', 231: 'Kanishk Seth', 232: 'Karan Sharma', 233: 'Karanveer Singh', 234: 'Karn Sharma', 235: 'Kartik Tyagi', 236: 'Karun Nair', 237: 'Kedar Jadhav', 238: 'Keemo Paul', 239: 'Kevin Pietersen', 240: 'Kevon Cooper', 241: 'Kieron Pollard', 242: 'Kishore Pramod Kamath', 243: 'Kona Srikar Bharat', 244: 'Krishnappa Gowtham', 245: 'Krismar Santokie', 246: 'Krunal Pandya', 247: 'Kshitiz Sharma', 248: 'Kuldeep Sen', 249: 'Kuldeep Yadav', 250: 'Kuldip Yadav', 251: 'Kulwant Khejroliya', 252: 'Kusal Janith Perera', 253: 'Kyle Abbott', 254: 'Kyle Jamieson', 255: 'Kyle Mayers', 256: 'Lakshmipathy Balaji', 257: 'Lalit Yadav', 258: 'Lasith Malinga', 259: 'Laxmi Ratan Shukla', 260: 'Liam Livingstone', 261: 'Lockie Ferguson', 262: 'Luke Pomersbach', 263: 'Lukman Hussain Meriwala', 264: 'Lungisani Ngidi', 265: 'Luvnith Sisodia', 266: 'M Siddharth', 267: 'M. Ashwin', 268: 'M. Harisankar Reddy', 269: 'Maheesh Theekshana', 270: 'Mahipal Lomror', 271: 'Manan Ajay Sharma', 272: 'Manan Vohra'}

my_dict3 = {273: 'Mandeep Hardev Singh', 274: 'Mandeep Singh', 275: 'Manish Pandey', 276: 'Manjot Kalra', 277: 'Manoj Tiwary', 278: 'Manpreet Gony', 279: 'Manprit Juneja', 280: 'Manvinder Bisla', 281: 'Manzoor Dar', 282: 'Marchant De Lange', 283: 'Marco Jansen', 284: 'Marcus Stoinis', 285: 'Mark Wood', 286: 'Martin Guptill', 287: 'Matt Henry', 288: 'Matthew Wade', 289: 'Mayank Agarwal', 290: 'Mayank Dagar', 291: 'Mayank Markande', 292: 'Mayank Yadav', 293: 'Michael Clarke', 294: 'Michael Hussey', 295: 'Midhun S', 296: 'Milind Kumar', 297: 'Milind Tandon', 298: 'Mitchell Johnson', 299: 'Mitchell Marsh', 300: 'Mitchell McClenaghan', 301: 'Mitchell Santner', 302: 'Mitchell Starc', 303: 'Mithun Manhas', 304: 'Moeen Ali', 305: 'Mohammad Nabi', 306: 'Mohammad Shami', 307: 'Mohammed Azharudeen', 308: 'Mohammed Siraj', 309: 'Mohd. Arshad Khan', 310: 'Mohit Sharma', 311: 'Mohsin Khan', 312: 'Moises Henriques', 313: 'Monu Singh', 314: 'Morne Morkel', 315: 'Mujeeb Zadran', 316: 'Mukesh Choudhary', 317: 'Munaf Patel', 318: 'Murali Kartik', 319: 'Murali Vijay', 320: 'Murugan Ashwin', 321: 'Mustafizur Rahman', 322: 'Muttiah Muralitharan', 323: 'N Jagadeesan', 324: 'N. Jagadeesan', 325: 'N. Tilak Varma', 326: 'Naman Ojha', 327: 'Nathan Coulter-Nile', 328: 'Nathan Ellis', 329: 'Nathan McCullum', 330: 'Nathu Singh', 331: 'Navdeep Saini', 332: 'Nic Maddinson', 333: 'Nicolas Pooran', 334: 'Nidheesh M D Dinesan', 335: 'Nikhil Shankar Naik', 336: 'Nitish Rana', 337: 'Noor Ahmad', 338: 'Obed Mccoy', 339: 'Odean Smith', 340: 'Oshane Thomas', 341: 'Pankaj Jaswal', 342: 'Pankaj Singh', 343: 'Paras Dogra', 344: 'Pardeep Sahu', 345: 'Parthiv Patel', 346: 'Parveez Rasool', 347: 'Parvinder Awana', 348: 'Pat Cummins', 349: 'Pavan Deshpande', 350: 'Pawan Negi', 351: 'Pawan Suyal', 352: 'Peter Handscomb', 353: 'Philip Hughes', 354: 'Piyush Chawla', 355: 'Prabhsimran Singh', 356: 'Pradeep Sangwan', 357: 'Pragyan Ojha', 358: 'Prasanth Padmanabhan', 359: 'Prasanth Parameswaran', 360: 'Prashant Chopra', 361: 'Prashant Solanki', 362: 'Prasidh Krishna', 363: 'Pratham Singh', 364: 'Pratyush Singh', 365: 'Praveen Dubey', 366: 'Praveen Kumar', 367: 'Pravin Dubey', 368: 'Pravin Tambe', 369: 'Prayas Ray Barman', 370: 'Prerak Mankad', 371: 'Prince Balwant Rai Singh', 372: 'Prithvi Raj Yarra', 373: 'Prithvi Shaw', 374: 'Priyam Garg', 375: 'Quinton De Kock', 376: 'R Samarth', 377: 'R. Ashwin', 378: 'R. Sai Kishore', 379: 'R. Sanjay Yadav', 380: 'Rahul Ajay Tripathi', 381: 'Rahul Buddhi', 382: 'Rahul Chahar', 383: 'Rahul Sharma', 384: 'Rahul Shukla', 385: 'Rahul Tewatia', 386: 'Rahul Tripathi', 387: 'Raj Angad Bawa', 388: 'Rajagopal Sathish', 389: 'Rajat Bhatia', 390: 'Rajat Patidar', 391: 'Rajvardhan Hangargekar', 392: 'Ramandeep Singh', 393: 'Ramesh Kumar', 394: 'Ranganath Vinay Kumar', 395: 'Rashid Khan Arman', 396: 'Rasikh Dar', 397: 'Rassie Van Der Dussen', 398: 'Ravi Bishnoi', 399: 'Ravi Bopara', 400: 'Ravi Rampaul', 401: 'Ravichandran Ashwin', 402: 'Ricky Bhui', 403: 'Ricky Ponting', 404: 'Riley Meredith', 405: 'Rinku Singh', 406: 'Ripal Patel', 407: 'Rishabh Pant', 408: 'Rishi Dhawan'}

my_dict4 = {409: 'Riyan Parag', 410: 'Robin Uthappa', 411: 'Romario Shepherd', 412: 'Ronit More', 413: 'Ross Taylor', 414: 'Rovman Powell', 415: 'Rudra Pratap Singh', 416: 'Ruturaj Gaikwad', 417: 'Ryan McLaren', 418: 'Ryan Ten Doeschate', 419: 'Sachin Baby', 420: 'Sachin Rana', 421: 'Sachithra Senanayaka', 422: 'Sagar Trivedi', 423: 'Sam Billings', 424: 'Sam Curran', 425: 'Samuel Badree', 426: 'Sandeep Bavanaka', 427: 'Sandeep Lamichhane', 428: 'Sandeep Sharma', 429: 'Sandeep Warrier', 430: 'Sanjay Yadav', 431: 'Sanju Samson', 432: 'Sarabjit Ladda', 433: 'Sarfaraz Khan', 434: 'Sarfaraz Naushad Khan', 435: 'Saurabh Dubey', 436: 'Saurabh Kumar', 437: 'Saurabh Tiwary', 438: 'Sayan Ghosh', 439: 'Sayan Sekhar Mandal', 440: 'Scott Boland', 441: 'Sean Abbott', 442: 'Shadab Jakati', 443: 'Shahbaz Ahamad', 444: 'Shahbaz Nadeem', 445: 'Shahrukh Khan', 446: 'Shakib Al Hasan', 447: 'Shane Watson', 448: 'Sharad Lumba', 449: 'Shardul Thakur', 450: 'Shashank Singh', 451: 'Shaun Marsh', 452: 'Sheldon Cottrell', 453: 'Sheldon Jackson', 454: 'Shelley Shaurya', 455: 'Sherfane Rutherford', 456: 'Shikhar Dhawan', 457: 'Shimron Hetmyer', 458: 'Shishir Bhavane', 459: 'Shivam Dube', 460: 'Shivam Mavi', 461: 'Shivam Sharma', 462: 'Shivil Kaushik', 463: 'Shreevats Goswami', 464: 'Shreyas Gopal', 465: 'Shreyas Iyer', 466: 'Shrikant Mundhe', 467: 'Shubam Agrawal', 468: 'Shubham Garhwal', 469: 'Shubham Ranjane', 470: 'Shubman Gill', 471: 'Siddarth Kaul', 472: 'Siddharth Kaul', 473: 'Siddhesh Dinesh Lad', 474: 'Simarjeet Singh', 475: 'Srikkanth Anirudha', 476: 'Steven Smith', 477: 'Stuart Binny', 478: 'Subhranshu Senapati', 479: 'Subramaniam Badrinath', 480: 'Sudeep Tyagi', 481: 'Sumit Narwal', 482: 'Suryakumar Yadav', 483: 'Sushant Marathe', 484: 'Suyash Prabhudesai', 485: 'Suyash Prabhudessai', 486: 'Swapnil Singh', 487: 'Syed Khaleel Ahmed', 488: 'Syed Mehdi Hasan', 489: 'T Natarajan', 490: 'T. Natarajan', 491: 'Tajinder Dhillon', 492: 'Tanmay Agarwal', 493: 'Tanmay Mishra', 494: 'Tejas Baroka', 495: 'Tejas Singh Baroka', 496: 'Thisara Perera', 497: 'Tim David', 498: 'Tim Seifert', 499: 'Tim Southee', 500: 'Tirumalasetti Suman', 501: 'Tom Banton', 502: 'Tom Curran', 503: 'Travis Head', 504: 'Trent Boult', 505: 'Tushar Deshpande', 506: 'Tymal Mills', 507: 'Umang Sharma', 508: 'Umesh Yadav', 509: 'Unmukt Chand', 510: 'Utkarsh Singh', 511: 'Vaibhav Arora', 512: 'Vaibhav Rawal', 513: 'Varun Aaron', 514: 'Varun Chakaravarthy', 515: 'Veer Pratap Singh', 516: 'Venkatesh Iyer', 517: 'Venugopal Rao', 518: 'Vicky Ostwal', 519: 'Vijay Shankar', 520: 'Vijay Zol', 521: 'Vikas Tokas', 522: 'Vikramjeet Malik', 523: 'Virat Singh', 524: 'Virender Sehwag', 525: 'Vishnu Vinod', 526: 'Wanindu Hasaranga', 527: 'Washington Sundar', 528: 'Wayne Parnell', 529: 'Wriddhiman Saha', 530: 'Writtick Chatterjee', 531: 'Yash Dayal', 532: 'Yash Dhull', 533: 'Yashasvi Jaiswal', 534: 'Yogesh Gowalkar', 535: 'Yogesh Takawale', 536: 'Yudhvir Charak', 537: 'Yusuf Pathan', 538: 'Yuvraj Singh', 539: 'Yuzvendra Chahal', 540: 'Yuzvendra Singh Chahal', 541: 'Zaheer Khan', 542: 'Zahir Khan Pakteen'}

my_dict5 =  {0: 'Chennai Super Kings', 1: 'Delhi Capitals', 2: 'Delhi Daredevils', 3: 'Gujarat Lions', 4: 'Gujarat Titans', 5: 'Kings XI Punjab', 6: 'Kolkata Knight Riders', 7: 'Lucknow Super Giants'}

my_dict6 =   {0: 'All-Rounder', 1: 'Batsman', 2: 'Bowler', 3: 'Wicket Keeper'}

my_dict7 =   {0: 'Indian', 1: 'Overseas'}

my_dict8 = {8: 'Mumbai Indians', 9: 'Pune Warriors India', 10: 'Punjab Kings', 11: 'Rajasthan Royals', 12: 'Rising Pune Supergiant', 13: 'Royal Challengers Bangalore', 14: 'Sunrisers Hyderabad'}

st.write("  \n")


# Display the dictionary in a table
st.markdown("#### Player Names and their Key Numbers :-")

col1, col2, col3, col4 = st.columns([1, 1, 1, 1])

col1, col2, col3, col4 = st.columns(4)

# Display content in each column
with col1:
  display_dictionary_in_table(my_dict1)

with col2:
  display_dictionary_in_table(my_dict2)

with col3:
  display_dictionary_in_table(my_dict3)

with col4:
  display_dictionary_in_table(my_dict4)

st.write("  \n")

st.markdown("#### Team Names and their Key Numbers :-")

col1, col2 = st.columns([1, 1])

col1, col2= st.columns(2)

# Display content in each column
with col1:
  display_dictionary_in_table(my_dict5)

with col2:
  display_dictionary_in_table(my_dict8)

st.write("  \n")

st.markdown("#### Roles and their Key Numbers :-")

display_dictionary_in_table(my_dict6)

st.write("  \n")


st.markdown("#### Origin and Key Number :-")

display_dictionary_in_table(my_dict7)

st.write("  \n")
st.write("  \n")

st.markdown("## Enter the player data here to predict the Salary :-")

# FUNCTION
def user_report():
  Player = st.text_input("Enter Player Key Number", )
  Role = st.slider('Enter Role Key Number', 0,3,0)

  Team = st.text_input("Enter Team Key Number", )
  Year = st.selectbox(
    'Select a Year',
    ('2025', '2026', '2027')
     )  
 
  Player_Origin = st.slider('Enter Player Origin', 0,1,0 )
  

  user_report_data = {
      'Player':Player,
      'Role':Role,

      'Team':Team,
      'Year':Year,
      'Player_Origin':Player_Origin,
        }
  report_data = pd.DataFrame(user_report_data, index=[0])
  return report_data

user_data = user_report()
st.subheader('Player Data')
st.write(user_data)

salary = model.predict(user_data)
st.subheader('Player Salary :-')
st.subheader('₹ '+str(np.round(salary[0], 2)))




