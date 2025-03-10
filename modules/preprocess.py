import sys,os
import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore',category=DeprecationWarning)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from nltk.corpus import stopwords,wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer,WordNetLemmatizer
import nltk

for PP in ['corpora/stopwords','corpora/wordnet','tokenizers/punkt_tab','taggers/averaged_perceptron_tagger']:
    try:
        nltk.data.find(PP)
    except LookupError:
        nltk.download(PP.split('/')[1])

    
class Preprocess_Text():
    
    def __init__(self,single_str_text,json=False,training=False,params=None):
        self.data = single_str_text
        self.json = json
        self.training = training
        self.params = params
        
    def get_originalText(self):
        """
        Description: Returns text with no changes
        Inputs: None
        Output: 
            self.data: a string with no changes
        """
        return self.data
    
    def convert_upper2lower(self,data):
        """
        Description: Converts all upper case letters to lower case
        Inputs: 
            data: the string to be processed
        Output: 
            self.data: string with all lower case values
        """
        return np.char.lower(data)
    
    def remove_stop_words(self,data,add_lst_stop_words=[],minus_lst_stop_words=[]):
        """
        Description: Removes list of all stop words in text
        Inputs: 
            data: the string to be processed
            add_lst_stop_words: words to be added to the list of stop words
            minus_lst_stop_words: words to be subtracted from the list of stop words
        Output: 
            new_text: string with all stop words removed
        """
        stop_words = stopwords.words('english')
        stop_words = list(set(stop_words+add_lst_stop_words)-set(minus_lst_stop_words))
        
        words = word_tokenize(str(data))
        new_text = ""
        for w in words:
            if w not in stop_words and len(w)>1:
                new_text = new_text + " " + w
        return new_text
    
    def remove_incorrect_words(self,data):
        """
        Description: Remove words that are mispelled from the text
        Inputs: 
            data: the string to be processed
        Output: 
            new_text: string with all mispelled words removed
        """
        words = word_tokenize(str(data))
        new_text = ""
        enchant_dict = enchant.Dict("en_US")
        
        for w in words:
            if enchant_dict.check(w) and len(w)>1:
                new_text = new_text + " "+w
        return new_text
        
    def spell_correction(self,data):
        """
        Description: For mispelled words, replace with top suggested word from enchant module
        Inputs: 
            data: the string to be processed
        Output: 
            new_text: string with mispelled words replaced with suggested correct word
        """
        words = word_tokenize(str(data))
        new_text = ""
        enchant_dict = enchant.Dict("en_US")
        
        for w in words:
            if len(w)>1:
                if enchant_dict.check(w):
                    new_text = new_text+ " "+w
                else:
                    try:
                        new_text = new_text+" "+enchant_dict.suggest(w)[0]
                    except:
                        new_text = new_text+" "+w
        return new_text
    
    def spell_correction1(self,data):
        """
        Description: For mispelled words, replace with top suggested word from enchant module
        Inputs: 
            data: the string to be processed
        Output: 
            new_text: string with mispelled words replaced with suggested correct word
        """
        words = word_tokenize(str(data))
        new_text = ""
        enchant_dict = enchant.Dict("en_US")
        
        for w in words:
            if len(w)>1:
                if enchant_dict.check(w):
                    new_text = new_text+ " "+w
                else:
                    spell = SpellChecker()
                    mispelled = list(spell.unknown([w]))
                    new_text = new_text+" "+spell.correction(mispelled[0])
        return new_text
    
    def remove_punctuations(self,data):
        """
        Description: Remove all punctuations and symbols from specified list
        Inputs: 
            data: the string to be processed
        Output: 
            new_text: string with all punctuations removed
        """
        symbols = "!\"#$%&()*+-./:;<=>?@[\]^_`{|}~\n"
        for i in range(len(symbols)):
            data = np.char.replace(data,symbols[i],' ')
            data = np.char.replace(data," "," ")
        data = np.char.replace(data,',','')
        return data
    
    def remove_apostrophe(self,data):
        """
        Description: Remove all apostrophes from text
        Inputs: 
            data: the string to be processed
        Output: 
            new_text: string with all apostrophes removed
        """
        return np.char.replace(data,"'"," ")
    
    def stemming(self,data):
        """
        Description: Replace all words with their root forms. Ex: 'connection' and 'connects' -> 'connect'
        Inputs: 
            data: the string to be processed
        Output: 
            new_text: string with all words stemmed
        """
        stemmer = PorterStemmer()
        tokens = word_tokenize(str(data))
        new_text = ""
        
        for w in tokens:
            new_text = new_text+" "+stemmer.stem(w)
        return new_text
    
    def lemmitization(self,data):
        """
        Description: Replace all words with their dictionary forms regardless of inflection or tense.
        Inputs: 
            data: the string to be processed
        Output: 
            new_text: string with all words lemmatized
        """
        lemm = WordNetLemmatizer()
        tokens = word_tokenize(str(data))
        new_text = ""
        
        for w in tokens:
            new_text = new_text+" "+lemm.lemmatize(w)
        return new_text
    
    def convert_num2txt(self,data):
        """
        Description: Convert integers in the text to text. Ex: 2 -> 'two'
        Inputs: 
            data: the string to be processed
        Output: 
            new_text: string with all numbers converted to text
        """
        tokens = word_tokenize(str(data))
        new_text = ""
        
        for w in tokens:
            try:
                w = num2words(np.int(w))
            except:
                a=0
            new_text = new_text+" "+w
        new_text = np.char.replace(new_text,"-"," ")
        return new_text
    
    def remove_numbers(self,data):
        """
        Description: Remove all integers from text
        Inputs: 
            data: the string to be processed
        Output: 
            new_text: string with all integers removed
        """
        data = str(data)
        return ''.join([i for i in data if not i.isdigit()])
    
    def preprocessText_pipeline(self):
        """
        Description: Complete processing steps based on specified config values
        Inputs: 
            json_input, upper2lower, remove_stop_words, 
            remove_incorrect_words, spell_correction,
            remove_punctuations, remove_apostrophe,
            remove_number, convert_numbers, lemmitization, stemming: Boolean values to determine if we run processing function
        Output: 
            str(self.data): Final processed output string with processing functions applied
        """
        if self.training:
            json_input = self.params['json_input']
            upper2lower = self.params['upper2lower']
            remove_stop_words = self.params['remove_stop_words']
            remove_incorrect_words = self.params['remove_incorrect_words']
            spell_correction = self.params['spell_correction']
            remove_punctuations = self.params['remove_punctuations']
            remove_apostrophe = self.params['remove_apostrophe']
            remove_numbers = self.params['remove_numbers']
            convert_numbers = self.params['convert_numbers']
            lemmitization = self.params['lemmitization']
            stemming = self.params['stemming']
            add_lst_stop_words = self.params['add_lst_stop_words']
            minus_lst_stop_words = self.params['minus_lst_stop_words']
        else:
            json_input = config['Preprocessing']['json_input']
            upper2lower = config['Preprocessing']['upper2lower']
            remove_stop_words = config['Preprocessing']['remove_stop_words']
            remove_incorrect_words = config['Preprocessing']['remove_incorrect_words']
            spell_correction = config['Preprocessing']['spell_correction']
            remove_punctuations = config['Preprocessing']['remove_punctuations']
            remove_apostrophe = config['Preprocessing']['remove_apostrophe']
            remove_numbers = config['Preprocessing']['remove_numbers']
            convert_numbers = config['Preprocessing']['convert_numbers']
            lemmitization = config['Preprocessing']['lemmitization']
            stemming = config['Preprocessing']['stemming']
            add_lst_stop_words = config['Preprocessing']['add_lst_stop_words']
            minus_lst_stop_words = config['Preprocessing']['minus_lst_stop_words']

            
        if json_input:
            self.data = extract_values(self.data,'text')
        
        if upper2lower:
            self.data = self.convert_upper2lower(self.data)
        
        if remove_numbers:
            self.data = self.remove_numbers(self.data)
            
        if remove_punctuations:
            self.data = self.remove_punctuations(self.data)
            
        if remove_stop_words:
            self.data = self.remove_stop_words(self.data,add_lst_stop_words,minus_lst_stop_words)
            
        if remove_apostrophe:
            self.data = self.remove_apostrophe(self.data)
            
        if spell_correction:
            self.data = self.spell_correction(self.data)
            
        if convert_numbers:
            self.data = self.convert_numbers(self.data)
            
        if remove_incorrect_words:
            self.data = self.remove_incorrect_words(self.data)
        
        if lemmitization:
            self.data = self.lemmitization(self.data)
        
        if stemming:
            self.data = self.stemming(self.data)
        
        self.data = ' '.join(self.data.split())
        return str(self.data)

    
def preprocess(data,training=False,params=None):
    """
    Description: Run preprocessing steps on given string
    Inputs:
        data: string we want to be processed
    Output: 
        PP.preprocessText_pipeline(): Final processed output string with processing functions applied
    """
    PP = Preprocess_Text(data,training=training,params=params)
    return PP.preprocessText_pipeline()