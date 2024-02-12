import pyttsx3
from sklearn.semi_supervised import LabelSpreading
import speech_recognition as sr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
import nltk
from sklearn.model_selection import train_test_split
import random
import warnings
warnings.simplefilter('ignore')
from sklearn.metrics import accuracy_score

#nltk.download("punkt")

def speak(text):
    engine = pyttsx3.init()
    voices = engine.getProperty('voices')
    ID = r'HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Speech\Voices\Tokens\TTS_MS_EN-US_ZIRA_11.0'
    engine.setProperty('voice' , ID)
    print("")
    print(f"==>Jarvis : {text}")
    print("")
    engine.say(text=text)
    engine.runAndWait()

def speechrecognitions():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening......")
        r.pause_threshold = 1
        audio = r.listen(source , timeout = 8)

    try:
        print("Recognizing....")
        query = r.recognize_google(audio , language="en") # type: ignore
        return query.lower()
    except:
        return ""

def Brain(query):
    Query = str(query).lower()
    if "house" in Query:
        speak("moye moye")
    elif "future" in Query:
        speak("vartman mein jeena sikho bachha")
    elif "crore" in Query:
        speak("Tera baap yaha chhod kar gaya ya yeri maa")
    elif "poty" in Query:
        speak("aap hi pottty khalo")
    elif "wife" in Query:
        speak("apni shakal dekhi hai tondu")
    elif "workless" in Query:
        speak("jis motu ne bola hai wahi hai")
    elif "motherfucker" in Query:
        speak("behen ke lode")
from datetime import date, datetime
time = datetime.now().strftime("%H:%M")

intents = {

            "greeting" : {
            "patterns": ["hi", "hello", "hey", "good morning", "good afternoon"],
            "responses": ["Hello there!", "Hi!", "Hey there!", "Good to see you!"]
        },
        
            "farewell" : {
            "patterns": ["bye", "goodbye", "see you", "later"],
            "responses": ["Goodbye!", "See you later!", "Have a nice day!"]
        },
        
            "thanks" : {
            "patterns": ["thanks", "thank you", "appreciate it"],
            "responses": ["You're welcome!", "Happy to help!", "Anytime!"]
        },
        
            "name_query" : {
            "patterns": ["what's your name", "who are you", "tell me your name"],
            "responses": ["I'm Jarvis, your virtual assistant.", "They call me Jarvis.", "I am Jarvis, at your service."]
        },
        
            "how_are_you" : {
            "patterns": ["how are you", "how's it going", "how do you do"],
            "responses": ["I'm doing well, thank you!", "I'm virtual, so no bad days!", "Couldn't be better!"]
        },
        
            "help" : {
            "patterns": ["can you help me", "I need assistance", "help me out"],
            "responses": ["Sure, what do you need help with?", "How can I assist you today?", "I'm here to help, what do you need?"]
        },
        
            "joke" : {
            "patterns": ["tell me a joke", "make me laugh", "I want to hear a joke"],
            "responses": ["Why don't scientists trust atoms? Because they make up everything!", "Why did the bicycle fall over? Because it was two-tired!", "What do you call fake spaghetti? An impasta!"]
        },
        
            "weather" : {
            "patterns": ["what's the weather like", "weather update", "is it raining"],
            "responses": ["I'm not currently able to check real-time weather, sorry!", "Please check a weather website for the latest updates.", "I'd recommend looking out the window or checking a weather app!"]
       
            },
    
        "date_time" : {
        "patterns": ["what day is it", "what's the time", "tell me the date"],
        "responses": [time]
    },
    
        "advice" : {
        "patterns": ["give me advice", "any tips", "what should I do"],
        "responses": ["Always think positively.", "Keep a balanced life.", "Remember to take breaks and relax."]
    },
    
        "fun_fact" : {
        "patterns": ["tell me a fun fact", "I want to hear a fun fact", "do you know any fun facts"],
        "responses": ["Did you know that honey never spoils?", "The shortest war in history was between Britain and Zanzibar on August 27, 1896. Zanzibar surrendered after 38 minutes.", "Did you know that Octopuses have three hearts?"]
    },
    
        "compliment" : {
        "patterns": ["say something nice", "give me a compliment", "can you compliment me"],
        "responses": ["You're doing great!", "You have an amazing spirit.", "You bring so much light to those around you."]
    },
    
        "general_query" : {
        "patterns": ["I have a question", "I need information", "Can you answer this"],
        "responses": ["I'll try my best to help. What's your question?", "I'm here to help, ask away.", "Sure, what do you want to know?"]
    },
    
        "music_recommendation" : {
        "patterns": ["recommend some music", "suggest a song", "what music do you like"],
        "responses": ["How about listening to 'Blinding Lights' by The Weeknd?", "I recommend 'Shape of You' by Ed Sheeran.", "You might enjoy 'Uptown Funk' by Mark Ronson."]
    },
        
        "technology" : {
        "patterns": ["tell me about technology", "latest tech news", "explain artificial intelligence"],
        "responses": ["Technology is rapidly advancing, bringing innovations in various fields.", "The latest tech news is available on tech news websites like TechCrunch or The Verge.", "Artificial Intelligence (AI) is the simulation of human intelligence in machines, often involving learning and problem-solving."]
    },
        
        "movie_recommendation" : {
        "patterns": ["suggest a movie", "what's a good film to watch", "favorite movie"],
        "responses": ["I recommend watching 'The Shawshank Redemption'.", "Consider watching 'Inception' for a mind-bending experience.", "If you like comedy, try 'The Grand Budapest Hotel'."]
    },
        
        "book_recommendation" : {
        "patterns": ["recommend a book", "what should I read", "favorite book"],
        "responses": ["I suggest reading 'The Hitchhiker's Guide to the Galaxy' by Douglas Adams.", "Consider 'To Kill a Mockingbird' by Harper Lee for a classic.", "If you enjoy fantasy, 'The Hobbit' by J.R.R. Tolkien is a great choice."]
    },
        
        "programming_help" : {
        "patterns": ["coding help", "programming assistance", "debug my code"],
        "responses": ["Sure, I can help with coding! Please share your code or describe the issue.", "I'll do my best to assist you with programming. What problem are you facing?", "Need help with a specific programming language or concept? Let me know."]
    },
        
        "motivation" : {
        "patterns": ["I need motivation", "feeling demotivated", "inspire me"],
        "responses": ["You've got this! Remember, every step forward is a step in the right direction.", "Challenges are what make life interesting. Overcoming them is what makes life meaningful.", "Believe in yourself, and all that you are. Know that there is something inside you that is greater than any obstacle."]
    },
        
        "travel_recommendation" : {
        "patterns": ["recommend a travel destination", "where should I go on vacation", "best places to visit"],
        "responses": ["Consider exploring Kyoto, Japan, known for its rich cultural heritage.", "If you love beaches, Maldives offers stunning turquoise waters and white sandy beaches.", "Prague, Czech Republic, is a beautiful city with a mix of history and architecture."]
    },
    
        "health_tip" : {
        "patterns": ["health advice", "tips for staying healthy", "how to stay fit"],
        "responses": ["Stay hydrated, drink plenty of water throughout the day.", "Regular exercise, even a short walk, can positively impact your health.", "A balanced diet with a variety of fruits and vegetables is essential for overall well-being."]
    },
        
        "science_facts" : {
        "patterns": ["tell me a science fact", "interesting science information", "science trivia"],
        "responses": ["Honey never spoils; archaeologists have found pots of honey in ancient Egyptian tombs that are over 3,000 years old.", "The total weight of all the ants on Earth is comparable to that of all the humans on Earth.", "Bananas are berries, but strawberries aren't!"]
    },
       "technology_news" : {
            "patterns": ["latest tech updates", "tell me about new gadgets", "tech advancements"],
            "responses": ["For the latest tech updates, visit websites like TechCrunch or Wired.", "Explore new gadgets on websites like The Verge or Gizmodo.", "Stay informed about tech advancements by following reputable tech news sources."]
        },
        
            "pet_advice" : {
            "patterns": ["pet care tips", "advice for pet owners", "how to train a dog"],
            "responses": ["Ensure your pets have a balanced diet and regular exercise for their well-being.", "Positive reinforcement is effective in training dogs. Reward good behavior!", "Regular veterinary check-ups are crucial for your pet's health."]
        },
        
            "inspirational_quote" : {
            "patterns": ["inspire me", "motivational quote", "give me inspiration"],
            "responses": ["The only way to do great work is to love what you do. - Steve Jobs", "Believe you can, and you're halfway there. - Theodore Roosevelt", "Don't watch the clock; do what it does. Keep going. - Sam Levenson"]
        },
        
            "coding_joke" : {
            "patterns": ["tell me a coding joke", "programming humor", "joke for developers"],
            "responses": ["Why do programmers prefer dark mode? Because light attracts bugs!", "Why do Java developers wear glasses? Because they don't see sharp!", "Why was the JavaScript developer sad? Because he didn't 'null' his feelings!"]
        },
        
            "learning_tip" : {
            "patterns": ["how to learn effectively", "study tips", "learning strategies"],
            "responses": ["Break your study sessions into smaller, focused intervals for better retention.", "Teach what you've learned to someone else; it reinforces your understanding.", "Use diverse resources—books, videos, and interactive content—to enhance your learning experience."]
        },
        
            "celebrity_news" : {
            "patterns": ["latest celebrity gossip", "what's happening in Hollywood", "celebrity updates"],
            "responses": ["Follow entertainment news websites like TMZ or E! Online for the latest celebrity updates.", "Stay tuned to social media platforms for real-time celebrity gossip.", "Explore magazines like People or Vanity Fair for exclusive celebrity interviews."]
        },
        
            "cooking_recipe" : {
            "patterns": ["suggest a recipe", "what to cook for dinner", "easy recipes"],
            "responses": ["How about trying a simple pasta carbonara for dinner tonight?", "Consider making a delicious homemade pizza with your favorite toppings.", "Explore a quick and healthy stir-fry recipe for a satisfying meal."]
        },
        
            "travel_tips" : {
            "patterns": ["travel advice", "tips for traveling", "packing tips"],
            "responses": ["Pack essential items like a travel adapter, comfortable shoes, and necessary medications.", "Research local customs and traditions before visiting a new destination.", "Create a checklist to ensure you don't forget important items while packing."]
        },
        
            "mindfulness" : {
            "patterns": ["mindfulness techniques", "how to practice mindfulness", "stress relief tips"],
            "responses": ["Practice deep breathing exercises to promote relaxation and reduce stress.", "Try mindfulness meditation to stay present in the moment and alleviate anxiety.", "Engage in activities like yoga or nature walks for a mindful and calming experience."]
        }
      
}

training_data = []
labels = []
for intent , data in intents.items():
    for pattern in data['patterns']:
        training_data.append(pattern.lower())
        labels.append(intent)


vectors = TfidfVectorizer(tokenizer=nltk.word_tokenize, stop_words="english" , min_df=1 , max_df=0.8)
training = vectors.fit_transform(training_data)

xtrain , xtest , ytrain , ytest = train_test_split(training , labels , test_size=0.4 , random_state=42 ,stratify=labels)

model = DecisionTreeClassifier()
model.fit(xtrain , ytrain)

predictions = model.predict(xtest)

print(accuracy_score(ytest , predictions))


def predict_intent(user_input):
    user_input = user_input.lower()
    input_vector = vectors.transform([user_input])
    intent = model.predict(input_vector)[0]
    return intent

print("AI Assistant: ")
speak(" Hello! How can I help you?")
while True:
    user_input = speechrecognitions()
    if user_input.lower() == 'exit':
        print("AI Assistant: Goodbye!")
        break
    intent = predict_intent(user_input)
    if intent in intents:
        responses = intents[intent]['responses']
        response = random.choice(responses)
        speak(response)
    else:
        speak("AI Assistant : Sorry, I'm not sure how to respond to that")
