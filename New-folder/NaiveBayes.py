class NaiveBayesClassifier:
    def __init__(self):
        self.priors = {}  # Prior probabilities of each class
        self.label_word_count= {}   
        self.likelihoods = {}  # Likelihood probabilities of each feature given each class
        self.vocab=set()       # Vocabulary of all unique words 
    

    def train(self, train_file):
        # Define a string that contains all punctuation characters
        punctuation = '''!"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~'''
        # Calculate prior probabilities
        with open(train_file, "r") as file:
            lines = file.readlines()
            label_counts = {}
            self.vocab=set()
            for line in lines:
                parts = line.strip().split(',', 1)
# Check if there are at least two parts after splitting
                if len(parts) != 2:
        # Skip this line if it doesn't contain a comma
                   continue

                label = parts[0].strip().strip('"').lower()  # Normalize label by removing double quotes and converting to lowercase
                if label in label_counts:
                    label_counts[label] += 1
                else:
                    label_counts[label] = 1

                text = parts[1] 
                words = text.split()
                 
                if label in self.label_word_count:
                    for word in words:
                        self.vocab.add(word)
                        if word in self.label_word_count[label]:
                            self.label_word_count[label][word] += 1
                        else:
                            self.label_word_count[label][word] = 2
                else:
                    self.label_word_count[label] = {}
                    for word in words:
                        if word in self.label_word_count[label]:
                            self.label_word_count[label][word] += 1
                        else:   
                            self.label_word_count[label][word] = 2

            total_label_count = len(lines)
            total_label_count = sum(label_counts.values())
        

        #calculate prior label probability 
        self.priors = {}
        for label, count in label_counts.items():
            self.priors[label] = count/total_label_count
        
        #print("vocabulary", len(self.vocab))

        

            
        
       
        
        


    def predict(self, test_file):
        #read the test file
        with open(test_file, "r") as file:
            lines = file.readlines()
            gold_labels = []
            predicted_labels = []
        
            #extract the gold lables from the test_file
            for line in lines:
                parts = line.strip().split(',', 1)
                # Check if there are at least two parts after splitting
                if len(parts) != 2:
                    # Skip this line if it doesn't contain a comma
                    continue
                label = parts[0].strip().strip('"')
                gold_labels.append(label)

                

                #analyse the sentence 
                text=parts[1]
                words = text.split()
                
                
                posteriors={}

                #adding unseen words 
                for label in self.label_word_count:
                    for word in words:
                        if word not in self.label_word_count[label]:
                            
                            self.label_word_count[label][word]=1

                #calculate likelihood probabilities with smoothing 
                
                for label in self.label_word_count:
                    self.likelihoods[label]={}
                    TotalWords=sum(self.label_word_count[label].values())
                    for word in self.label_word_count[label]:
                        WordlableCount=self.label_word_count[label][word]
                        self.likelihoods[label][word]=WordlableCount/TotalWords
                        
            

                all_labels=['joy','sadness','guilt','disgust','fear','shame','anger']
                
                
               
                for label in all_labels:
                    likelihoods=1
                    for word in words:
                        likelihoods=likelihoods*self.likelihoods[label][word]    
                    posteriors[label]=likelihoods*self.priors[label]
                
                             
                                
                #choose the label with maximum probability 
                max_key = max(posteriors, key=lambda k: posteriors[k])
                predicted_labels.append(max_key)
            
            return predicted_labels, gold_labels
    
    def evaluate(self, gold_labels, predicted_labels, target_label):
        tp = fp = fn = tn = 0

        for gold, pred in zip(gold_labels, predicted_labels):
            if gold == target_label and pred == target_label:
                tp += 1
            elif gold != target_label and pred == target_label:
                fp += 1
            elif gold == target_label and pred != target_label:
                fn += 1
            elif gold != target_label and pred != target_label:
                tn += 1

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'true_positives': tp,
            'false_positives': fp,
            'false_negatives': fn,
            'true_negatives': tn,
             'support': tp + fn  # Number of true instances for the target label
        }

    def evaluate_all_labels(self, gold_labels, predicted_labels, labels):
        evaluations = {}
        for label in labels:
            evaluations[label] = self.evaluate(gold_labels, predicted_labels, label)
        return evaluations
    
    def general_f1_score(self, evaluations):
        total_support = sum(evaluation['support'] for evaluation in evaluations.values())
        weighted_f1_score = sum(evaluation['f1_score'] * evaluation['support'] for evaluation in evaluations.values()) / total_support
        return weighted_f1_score
        
            
# Usage example
nb_classifier = NaiveBayesClassifier()
nb_classifier.train('isear-train.csv')

gold_labels, predicted_labels = nb_classifier.predict('isear-test.csv')
# print("Predictions:", predicted_labels)



# Evaluate performance for all labels
all_labels = ['joy', 'sadness', 'guilt', 'disgust', 'fear', 'shame', 'anger']
evaluations = nb_classifier.evaluate_all_labels(gold_labels, predicted_labels, all_labels)
for label, evaluation in evaluations.items():
    print(f"Evaluation for '{label}':", evaluation)      

# Calculate general F1-score
general_f1 = nb_classifier.general_f1_score(evaluations)
print("General F1-Score:", general_f1)

        
           
                

