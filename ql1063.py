import nltk
from nltk.tokenize import word_tokenize
import string
import nltk
import string
import math
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import defaultdict
from nltk.stem import PorterStemmer
stemmer = PorterStemmer()


### We use bigrams for this version and Porter Stemmer


def tokenize_queries(doc):
    queries = {}
    lines = doc.strip().split('\n')
    current_id = 0

    
    # Use the following stop words
    closed_class_stop_words = ['a','the','an','and','or','but','about','above','after','along','amid','among',\
                           'as','at','by','for','from','in','into','like','minus','near','of','off','on',\
                           'onto','out','over','past','per','plus','since','till','to','under','until','up',\
                           'via','vs','with','that','can','cannot','could','may','might','must',\
                           'need','ought','shall','should','will','would','have','had','has','having','be',\
                           'is','am','are','was','were','being','been','get','gets','got','gotten',\
                           'getting','seem','seeming','seems','seemed',\
                           'enough', 'both', 'all', 'your' 'those', 'this', 'these', \
                           'their', 'the', 'that', 'some', 'our', 'no', 'neither', 'my',\
                           'its', 'his' 'her', 'every', 'either', 'each', 'any', 'another',\
                           'an', 'a', 'just', 'mere', 'such', 'merely' 'right', 'no', 'not',\
                           'only', 'sheer', 'even', 'especially', 'namely', 'as', 'more',\
                           'most', 'less' 'least', 'so', 'enough', 'too', 'pretty', 'quite',\
                           'rather', 'somewhat', 'sufficiently' 'same', 'different', 'such',\
                           'when', 'why', 'where', 'how', 'what', 'who', 'whom', 'which',\
                           'whether', 'why', 'whose', 'if', 'anybody', 'anyone', 'anyplace', \
                           'anything', 'anytime' 'anywhere', 'everybody', 'everyday',\
                           'everyone', 'everyplace', 'everything' 'everywhere', 'whatever',\
                           'whenever', 'whereever', 'whichever', 'whoever', 'whomever' 'he',\
                           'him', 'his', 'her', 'she', 'it', 'they', 'them', 'its', 'their','theirs',\
                           'you','your','yours','me','my','mine','I','we','us','much','and/or'
                           ]

    
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith(".I"):
            # Use this to get the correct ID from the file
            current_id += 1
            i += 1
        elif line.startswith(".W"):
            i += 1
            query_lines = []
            while i < len(lines) and not lines[i].startswith(".I") and not lines[i].startswith(".W"):
                query_lines.append(lines[i])
                i += 1
            query_text = " ".join(query_lines)
            all_words = word_tokenize(query_text)
            non_stop = [word for word in all_words if word.lower() not in closed_class_stop_words]
            non_stop_no_punc = [word for word in non_stop if word not in string.punctuation]
            
            # Convert to bigram
            bigrams = list(nltk.bigrams(non_stop_no_punc))
            queries[current_id] = [(' '.join((stemmer.stem(word[0]), stemmer.stem(word[1])))) for word in bigrams]
        else:
            i += 1    
    return queries


def compute_idf(queries):
    N = len(queries)
    df = defaultdict(int)
    idf = defaultdict(float)
    for key in queries:
        for word in set(queries[key]):
            df[word] += 1

    for word, count in df.items():
        idf[word] = math.log(N / count)
    return idf

def compute_tf_idf(tokens_map, idf, term_freq_map=None):
    tf_idf = {}
    for key, tokens in tokens_map.items():
        # If term_freq_map is provided, use it; otherwise compute TF from scratch
        if term_freq_map and key in term_freq_map:
            tf = term_freq_map[key]
        else:
            tf = defaultdict(int)
            for word in tokens:
                tf[word] += 1

        total_terms = len(tokens)
        tf_idf[key] = {word: (math.log(1 + freq / float(total_terms))) * idf[word] for word, freq in tf.items()}

    return tf_idf

def tokenize_abstracts(doc):
    abstracts = {}
    lines = doc.strip().split('\n')
    current_id = None
    # Your stop_words list stays unchanged
    stop_words = ['a','the','an','and','or','but','about','above','after','along','amid','among',\
                           'as','at','by','for','from','in','into','like','minus','near','of','off','on',\
                           'onto','out','over','past','per','plus','since','till','to','under','until','up',\
                           'via','vs','with','that','can','cannot','could','may','might','must',\
                           'need','ought','shall','should','will','would','have','had','has','having','be',\
                           'is','am','are','was','were','being','been','get','gets','got','gotten',\
                           'getting','seem','seeming','seems','seemed',\
                           'enough', 'both', 'all', 'your' 'those', 'this', 'these', \
                           'their', 'the', 'that', 'some', 'our', 'no', 'neither', 'my',\
                           'its', 'his' 'her', 'every', 'either', 'each', 'any', 'another',\
                           'an', 'a', 'just', 'mere', 'such', 'merely' 'right', 'no', 'not',\
                           'only', 'sheer', 'even', 'especially', 'namely', 'as', 'more',\
                           'most', 'less' 'least', 'so', 'enough', 'too', 'pretty', 'quite',\
                           'rather', 'somewhat', 'sufficiently' 'same', 'different', 'such',\
                           'when', 'why', 'where', 'how', 'what', 'who', 'whom', 'which',\
                           'whether', 'why', 'whose', 'if', 'anybody', 'anyone', 'anyplace', \
                           'anything', 'anytime' 'anywhere', 'everybody', 'everyday',\
                           'everyone', 'everyplace', 'everything' 'everywhere', 'whatever',\
                           'whenever', 'whereever', 'whichever', 'whoever', 'whomever' 'he',\
                           'him', 'his', 'her', 'she', 'it', 'they', 'them', 'its', 'their','theirs',\
                           'you','your','yours','me','my','mine','I','we','us','much','and/or'
                           ]
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith(".I"):
            current_id = int(line.split()[1])
        elif line.startswith(".W"):
            abstract_lines = []
            i += 1
            while i < len(lines) and not lines[i].startswith("."):
                abstract_lines.append(lines[i])
                i += 1
            abstract_text = " ".join(abstract_lines)
            tokens = word_tokenize(abstract_text)
            # filtered_tokens = [word.lower() for word in tokens if word.lower() not in stop_words and word not in string.punctuation and not word.isdigit()]
            filtered_tokens = [stemmer.stem(word.lower()) for word in tokens if word.lower() not in stop_words and word not in string.punctuation and word.isdigit() == False]
            
            # Convert to bigram
            bigrams = list(nltk.bigrams(filtered_tokens))
            abstracts[current_id] = [(' '.join((stemmer.stem(word[0]), stemmer.stem(word[1])))) for word in bigrams]
            continue
        i += 1

    return abstracts
    
# Change path to the location of the cran.qry file
queries_path = '/Users/liqingyang/nyu_courses/natural_lang_processing/homeworks/hw4_Cranfield_collection/cran.qry'
with open(queries_path, "r") as file:
    doc = file.read()
queries = tokenize_queries(doc)
idf_queries = compute_idf(queries)

# Get the frequency as specified on the assignment and put that as a parameter in the compute_tf_idf function
queries_term_freq = {}
for q_id, tokens in queries.items():
    word_freq = defaultdict(int)
    for word in tokens:
        word_freq[word] += 1
    queries_term_freq[q_id] = word_freq

tf_idf_scores_queries = compute_tf_idf(queries, idf_queries, queries_term_freq)

# Change path to the location of the cran.all file
abstract_path = '/Users/liqingyang/nyu_courses/natural_lang_processing/homeworks/hw4_Cranfield_collection/cran.all.1400'
with open(abstract_path, "r") as file:
    abstracts_content = file.read()

abstracts = tokenize_abstracts(abstracts_content)
idf_abstract = compute_idf(abstracts)

# Cosine similarity function
def cosine_similarity(vector1, vector2):
    dot_product = sum(p*q for p,q in zip(vector1, vector2))
    magnitude = math.sqrt(sum([val**2 for val in vector1])) * math.sqrt(sum([val**2 for val in vector2]))
    if not magnitude:
        return 0
    return dot_product/magnitude

# Count number of instances of each non-stop-word in each abstract
abstract_term_freq = {}
for abstract_id, tokens in abstracts.items():
    word_freq = defaultdict(int)
    for word in tokens:
        word_freq[word] += 1
    abstract_term_freq[abstract_id] = word_freq

# Compute TF-IDF for abstracts
tf_idf_scores_abstracts = compute_tf_idf(abstracts, idf_abstract, abstract_term_freq)

# Function to get vector for comparison
def get_vector(query_terms, document):
    vector = [document.get(term, 0) for term in query_terms]
    return vector

# Compare queries with abstracts using cosine similarity
results = defaultdict(dict) # Store the results as {query_id: {abstract_id: cosine_similarity}}

for query_id, query in tf_idf_scores_queries.items():
    for abstract_id, abstract in tf_idf_scores_abstracts.items():
        query_vector = get_vector(query.keys(), query)
        abstract_vector = get_vector(query.keys(), abstract)
        
        similarity = cosine_similarity(query_vector, abstract_vector)
        if similarity > 0:
            results[query_id][abstract_id] = similarity

# Now, results contain cosine similarity scores between each query and abstract
sorted_results = {}

for query_id in results:
    sorted_abstracts = sorted(results[query_id].items(), key=lambda x: x[1], reverse=True)
    sorted_results[query_id] = [abstract[0] for abstract in sorted_abstracts]

# At this point, sorted_results will have for each query a ranking of abstracts.
# Sort abstracts for each query by their similarity scores in descending order
sorted_results = {query_id: sorted(abstracts.items(), key=lambda x: x[1], reverse=True) 
                  for query_id, abstracts in results.items()}

# Convert sorted results to desired format
formatted_results = []

for query_id, abstracts in sorted_results.items():
    for abstract_id, similarity in abstracts:
        formatted_results.append(f"{query_id} {abstract_id} {similarity:.9f}")

with open("output.txt", "w") as out_file:
    for line in formatted_results:
        out_file.write(line + "\n")
print("Done")