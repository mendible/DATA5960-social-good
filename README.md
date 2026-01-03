# Named Entity Recognition for Police Misconduct Incidents in Minneapolis 

## Introduction & Project Goals

This repository is part of an independent study project at Seattle University, mentored by Dr. Ariana Mendible. I worked with two other students in the M.S. in Data Science program: Jesse Loi and Mark Daza. 

As part of the Data Science, Police Accountability, Community Empowerment (DSPACE) group, Dr. Mendible did a research project on police misconduct in Minneapolis. This independent study is a continuation of that project. More specifically, her team conducted network analysis to understand how police misconduct is shaped by social dynamics between officers. In the network, each node represented an officer in the Minneapolis Police Department who had a complaint against them. Two officers had an edge between them if both were involved in the same complaint. 

According to Dr. Mendible, a limitation of the network analysis is that all complaints were treated equally, regardless of context. DSPACE didn't have access to information about the incident that happened in each complaint, who the victim was, whether force was used, the date or location of the incident, and so on. When outlining aims for this project, Dr. Mendible said that her next step was to gather more context and information about each police incident. In the data archive we used, some police officers have documents, like news articles or court cases, about complaints they were involved in. These documents contain the context we need.

Our project goal is to accurately extract the date and location of the police incident in each document. We are creating a pipeline that combines a large language model (LLM) with named entity recognition (NER) to extract that information from the documents. Combining the strengths of both models will create a more accurate information extraction pipeline.

A given document may mention several dates and locations. However, only one of these dates/locations is the date/location of the police incident, and the others may be irrelevant. Because an LLM can understand context, it can identify *which* of these dates/locations is the date/location of the police incident. However, an LLM can hallucinate and output information that is not in the text of the document. 

NER simply assigns a category (e.g.: "person", "organization", "date") to each entity detected in the text. Therefore, NER is guaranteed to output only information that's in the article. NER can identify all locations/dates in the text, but it cannot identify which of these is the location/date of the police incident. This is because NER cannot understand context like LLMs. Additionally, an LLM can identify whether there even *is* a police incident mentioned in the document, while NER cannot. 

This repository contains my portion of the project, which is an NER system that extracts all locations and dates mentioned in a document. In future work, this NER system will be combined with an LLM (Llama 3.1) to build our pipeline.

## Data

### Data source

DSPACE gathered data from the [Citizens United Against Police Brutality (CUAPB) police complaint archive](https://complaints.cuapb.org/police_archive/). This archive contains police complaints against officers throughout Minnesota. Each officer has a webpage listing any complaints against them, along with documents related to the complaint(s).

I obtained the raw dataset from Jesse Loi, who performed data scraping for this project. He scraped all the police complaint documents found in the archive. He provided me with a CSV file containing 15,527 rows with the following columns:
- `text`: the text in the police complaint document
- `name`: the name and badge number of the officer that the document is about
- `department`: the police department where that officer works
- `url`: the URL where the document is found

### Data preparation

I performed the following steps to clean and prepare the data:
- Drop the rows with no article text. These rows contained `NaN` or placeholders such as "No text found" or "Too_Long". This removed over 12,000 rows, which was most of the dataset.

- Investigate rows where the text starts with "Too_Long", but there is still article text. There were 20 such rows. These documents seemed to be cut off in the middle, which suggests that the scraper had a limit to the number of characters it can read. I kept these rows, and I kept the "Too_Long" to keep track of which articles are incomplete. These rows are only a small fraction of the whole dataset, so they do not significantly affect the dataset.

- Remove rows with invalid URLs
- Drop duplicate rows. There were 17 such rows.
- Manually inspect rows where the text seems to be invalid. Drop all texts that turn out to be invalid. "Invalid" texts are those that do not contain a police incident document. For example, some texts just said "Access denied". Others contained just the heading of a document, but none of the content. 

- Check if the values in the `name` column are valid. All were valid.
- Check the `department` column for invalid values. Some rows contained just "Department:". The other data in those rows was valid, so I replaced those departments with `NaN` to indicate a missing department. There were 9 rows with a missing department.

- The scraped text contained many "weird" Unicode characters that many NLP models cannot read. I attempted to convert as much of the text to ASCII as I could. First, I identified a few commonly occurring Unicode characters and manually replaced them with the appropriate ASCII character. For example, the Unicode "ffi" ligature (\uFBO3) often appeared in the common word "officer". I replaced this ligature with the ASCII characters "ffi". Next, I applied `ftfy`'s `fix_text()` function to the texts. That function fixes inconsistencies and "bad" characters in Unicode text. Now, many of the "weird" characters were gone, making the text more readable. However, many words containing the letters "tt", "ti", or "fi" were altered. For example, "compensation", "committee", and "settlement" were often changed to "compensaon", "commiee", and "selement". Many common words were now incorrectly spelled. This can present a major problem to NLP models, but it was a necessary step to improve the models' ability to read the overall text. Furthermore, some Unicode characters still remained in the text.

- Inspect duplicate texts. Out of 2,302 remaining rows, 908 texts were duplicates. There were 900 rows where the text *and* the URL were duplicates of another row. I determined that, if an article is about multiple police officers, then each officer gets a separate row, and all those rows contain the same article text. 
- To deduplicate the texts, I did the following: 
    1. Created one DataFrame containing each unique text and an integer ID corresponding to that text. Saved this DataFrame to `police_reports.csv`.
    2. Created a separate DataFrame (`metadata_df`) containing the text IDs and all metadata for each text. The metadata consisted of the `name`, `department`, and `url` columns. Each occurrence of the duplicate texts had a separate row in `metadata_df`, since each occurrence had different metadata values. Saved `metadata_df` to `police_report_metadata.csv`.
- In the final data, there were 1,374 unique texts and 2,236 rows in `police_report_metadata.csv`.

## NER Implementation

### Models

I used one NER model to tag dates and another to tag locations. Both models were from the Flair library in Python. For locations, I used the `Saisam/Inquirer_ner_loc` model from HuggingFace. This model was fine-tuned specifically to detect locations. The NER models that come with Flair can tag locations, but when I compared models, `Saisam/Inquirer_ner_loc` was much more accurate. For dates, I used the `flair/ner-english-ontonotes-large` model from HuggingFace. This model comes with Flair and can predict 18 tags, including `DATE`. 

### Chunking the text

Flair's NER models run predictions on a `Sentence` object, or a list of these objects. A `Sentence` object can contain a sentence from a document, a whole document, or an arbitrary chunk of text from a document (e.g.: a paragraph, a few sentences, a 50-word chunk, etc.). I compared the following approaches:
    - Running NER on a whole document. (i.e.: having one Sentence object for the whole document)
    - Splitting the document into smaller chunks and running NER on the list of chunks (i.e.: having a list of Sentence objects, where each object contains a chunk).

The second approach was more accurate, so it was necessary to chunk the text before passing it to the NER system. I implemented the following chunking strategies:
    - Creating a chunk for each sentence. I used the NLTK library to tokenize the text into sentences.
    - Creating a chunk every `n` words. (For example, `n` could be 50, 100, or 200).

I compared the NER results for chunking by sentences and chunking every `n` words, with various values of `n`. When tagging locations, every chunking strategy did better at predicting some locations, and worse at predicting others. There was no chunking strategy that was consistently the most accurate for *all* locations in a given text. For tagging dates, the predicted entities were almost the same for all chunking strategies. Since there was no "best" chunking strategy, I chose to chunk by sentences.

### Confidence scores

Flair NER models output a confidence score between 0 and 1 for each predicted entity. My code allows the user to set a threshold for the confidence score. For example, you can keep only the entities whose confidence score is above 0.8. 

### Post-processing

The predicted entities for both locations and dates required a great deal of post-processing, due to false positives and extra punctuation. 

#### Locations

The predicted location entities often contained extra trailing punctuation. Examples are "Canada." or "Robbinsdale,". The model would also output *generic* locations (e.g.: "bridge", "hospital", "psychiatric ward", etc.). We only want to tag *specific* locations, such as street, city, or state names. Additionally, many of the news article texts contained the URL of the article. If that URL contained a location name, the model often tagged the whole URL as an entity, which is undesirable. 

Therefore, I post-processed the predicted location entities by:
- Removing leading and trailing whitespace and punctuation from all entities
- Filtering out URLs using a regular expression (regex). (However, this regex did not successfully filter out *all* URLs).
- Attempting to filter out the generic locations using a heuristic. If an entity is all lowercase, I dropped it, because specific locations always start with an uppercase letter. However, this is a naive solution that doesn't always work, because generic locations can be uppercase too (e.g.: "U.S. airports").

#### Dates

The team agreed to keep only date entities that contain a *month, day, and year*. Additionally, I converted all date entities to `datetime.date` objects in YYYY-MM-DD format, rather than strings. A consistent date format will make the dates easier for the LLM to work with.

The NER model consistently detected all the *valid* dates in an article. However, it also tagged many "extra" entities, such as time durations, weekdays, and people's ages. Some examples are:
- "Wednesday"
- "eight years"
- "21-year-old"
- "last month"
- "weekly"
- Case numbers such as "2021-01180"

I performed the following steps to post-process the predicted date entities: 
- Used Pandas to attempt parsing each entity to a `datetime.date` object with a month, day, and year. 
- Dropped all entities that Pandas could not parse
- Checked if Pandas parsed the remaining entities correctly. If an entity does not contain a month, day, *and* year, then Pandas may fill in a default value for the missing component(s). For example, Pandas parses "2021" as "01/01/2021". That is a hallucination because "January 1st, 2021" did not appear in the document. Such hallucinations are a major problem because our motivation for NER is to only output information that's in the text. So, I used semantic validation to compare all `date` objects returned by Pandas with the original entity text. If the parsed `date` object didn't have the same month, day, and year as the original text entity, I dropped that entity.
- Returned all remaining `date` objects.

These steps successfully filtered out all the "extra" entities and removed the Pandas hallucinations.

## Preliminary Findings

In this section, I outline the general trends I saw after running the NER system on a small number of documents. The findings will be more qualitative than quantitative. Due to time constraints, I could not manually annotate the articles to get the ground truth dates and locations. Thus, I could not calculate metrics like precision and recall. 

Instead, I chose a small "test set" of 20 documents from the dataset, ensuring that they contain a wide variety of document types. The "test set" contained news articles, workers' compensation claims, other legal documents, and letters. I manually inspected the NER results, compared them to the document, and found general areas where NER performed well or poorly. Below, I describe the areas in which NER excels or struggles, and ideas for future improvement. I will refer to specific texts by their text IDs in the dataset. For example, "text #59" refers to the text whose ID is 59.

### Detecting Dates

#### High Accuracy across All Date Formats

NER performed exceptionally well at predicting dates. The NER system to predict dates was accurate and robust to edge cases, exce/pt for one important case where false positives occur. The system almost always detected all the valid dates in the article. With the exception of one case, it did not tag anything that's *not* a valid date. The documents contained a wide variety of date formats. Examples are "MM/DD/YY", "YYYY-MM-DD", dates with textual month names, dates with month abbreviations (e.g.: "Oct 20, 2021"), and many more. The system detected every possible date format equally well. Moreover, the system performed equally well across all *types* of texts, such as news articles, court cases, and workers' compensation claims.

Below, I provide concrete examples of entities inside documents that the system tagged (or did not tag) correctly:

The system correctly tagged the following entities as dates:
- "Oct 20, 2021"	
- "MARCH 7, 2013"
- "MAR 8, 2018"
- "January 23, 2012"
- "09/16/04"
- "12/4/2021"
- "2007/01/26"

The system correctly did *not* tag the following entities as dates: 
- "four years"
- "45-year-old"
- "Tuesday"
- "2021"
- "weekly"
- "the week of October 11-16, 2021"
- "today"
- "May 2010"

#### False Positives

False positives can occur when a news article contains its URL, and the URL contains the publication date. An example of these URLs is "https://www.mprnews.org/story/2007/01/26/financialtaskforce". The NER system tags the date inside that URL (in this case, "2007-01-26"), even if that date is not in the article text. In the timeframe of the project, this could not be solved with post-processing. The system did not tag the *whole* URL, only the text "2007/01/26". The current post-processing logic lets this text through because it has a month, day, and year. News article texts often contain extra information, such as the URL and the text on the webpage above or below the article. Therefore, this hallucination can happen often. Extra post-processing logic to filter out these URLs is required. For example, the logic can check if the predicted entity is contained inside another string. If this larger string is above a certain length (e.g.: 40 characters), it is likely to be a URL, so the entity is invalid. 

#### Confidence Scores

The confidence scores for dates were exceptionally high (almost always above 0.9, and often above 0.99)! Both valid and invalid dates had high confidence scores. The post-processing logic filters out all invalid dates, except for the false positives with URLs. The false positives still had high confidence scores, so a confidence threshold is ineffective at removing them. Therefore, I believe a confidence threshold for dates is unnecessary. 

#### Final Recommendation

After the URL hallucination is addressed, I have confidence that the NER system for predicting dates is accurate enough to be integrated into the pipeline. Based on the small number of NER results I've examined so far, I believe NER can almost always provide the LLM with the correct list of dates in any police document. However, the NER system should be tested more rigorously before I can truly make this conclusion. For example, I can use metrics like recall and precision to evaluate the NER system. I can use a larger "test set" of police documents (e.g.: 200 instead of 20). I can verify more rigorously that this "test set" contains all possible document types in the dataset (news articles, court cases, etc.).

### Detecting Locations

The results for tagging locations were poorer and more ambiguous. Predicting locations is much more complicated because there are far more words and phrases that could possibly be locations, than ones that could possibly be dates. 

#### High Accuracy across All Location Types

The NER system can identify a wide variety of location types with high accuracy. 

`Inquirer_ner_loc` (the fine-tuned model I used) performed far better than the general NER models that come with Flair. The general NER models could not detect certain address components, like street address numbers. They also detected the street, city, and state of an address as three separate entities. Finally, they couldn't detect intersections (e.g.: "W 7th St & Kellogg Blvd, St. Paul, MN") as one entity. In many of our documents, the location of the police incident is an address or an intersection. Thus, it's important for our NER system to correctly extract these as one entity.

Additionally, our police incident documents contain a wide variety of location types, including addresses, street names, building names, cities, counties, and states. Thus, our NER system must accurately extract any location type. The general NER models couldn't do this.

The NER system using `Inquirer_ner_loc` satisfies both of these requirements. It can identify all location types, no matter how fine-grained or broad, with equally high accuracy.

For example, the NER system correctly extracted the following locations as one entity, without removing parts of the location name, or adding extra words that are not part of the location:
- "350 S. 5th St. Minneapolis, MN 55415"
- "Itasca County"
- "Ward 9 Midtown Phillips Intersection of 14th Avenue South and 24th Street East"
- "Second and Third precincts"
- "900 block of St. Anthony Avenue"
- "Minneapolis"
- "St. Paul, Minn."
- "Minnesota"
- "Estes Funeral Chapel in Minneapolis"

#### Robust to text encoding issues

The NER system could identify locations even when words were spelled incorrectly due to the Unicode issues I discussed. For example, after text cleaning, some words ended up with missing letters, such as "aorney" instead of "attorney", "selement" instead of "settlement", and "commiee" instead of "committee". The system still identified location entities that contained the misspelled words. For example, it identified "City Aorney's Office" as a location (which should be "City Attorney's Office"). Furthermore, when a location entity was misspelled because it contained an incorrect Unicode character, the system could still identify that location. For example, one article contained the location "Ramsey County Attorney's Office", where the "fi" was replaced with a random Unicode character. The system still detected the entire location entity correctly. However, in both cases of misspelling, the predicted entity that the NER outputs is still misspelled.

#### Incomplete Locations

There were many instances when NER extracted only *part* of a location. For several addresses containing a street, city, state, and zip code; the system extracted only the first few components (the street and city, for example), instead of the entire address. Other times, it extracted two separate entities, each containing some components of the address. For example, text #59 contained the address "1600 University Avenue West, Suite 200, Saint Paul, Minnesota 551 04-3825". The system identified the following entities: "1600 University Avenue West" and "Saint Paul, Minnesota 551". Another example of this partial extraction was when the system tagged "Minnesota Bureau" or "Minnesota Bureau of Criminal", but the actual location entity was "Minnesota Bureau of Criminal Apprehension".

I believe this is because I chunked the text instead of keeping it as one document. The complete location entity may have spanned two different chunks. However, chunking was necessary because it drastically improved the model's overall performance. As an example, without chunking, the system detected "writers" as a location.
 
A potential solution is to add a post-processing step that checks the span of each entity and combines entities with consecutive spans. A span is defined as the starting position and ending position of a predicted entity. For example, suppose that NER tags the entity "Minnesota", and suppose the word "Minnesota" starts at character #343 of a text. Then the span of "Minnesota" would be (343, 351). Suppose character #352 is a whitespace. Now suppose NER tags "Bureau of Criminal Apprehension" as another entity, and the span of that entity starts at character #353. Then, we can check if the only characters separating "Minnesota" and "Bureau of Criminal Apprehension" are whitespaces. If so, then the two entities have consecutive spans, so they are likely to be one entity, and we can combine them. However, this may not always work. Unstructured text can contain many edge cases. There might be cases when two entities appear right next to each other, but are actually two separate locations.

#### Adding Extra Text to Entities

There were also cases when predicted entities contained "extra" words directly before or after the correct location. Examples include:
- "Minnesota's"
- "Ex-Itasca County" (in the sentence "Ex-Itasca County deputy pleads guilty.")
- "UNITED STATES DISTRICT COURT District of Minnesota Daniel" (from a court case whose title was "UNITED STATES DISTRICT COURT District of Minnesota Daniel L. Fancher")
- "Twin Cities Minneapolis" (from an article title: "MPR News Notes on the news from the Twin Cities Minneapolis cops lauded for arresting anti-Somali attacker")
- "MINNEAPOLIS Minneapolis" (not shown in the test set)
- "Sioux Falls City of Sioux Falls" (not shown in the test set)
- "Minnesota Weather" (not shown in the test set)
- "Minnesota since" 
- "Hennepin County Attorney" (not shown in the test set)

In some of these examples, a probable issue is that there is no newline character separating the heading of a document from its content. Thus, the model thinks "MINNEAPOLIS Minneapolis" (for example) is one location. In fact, it seems that "MINNEAPOLIS" is the end of the heading and "Minneapolis" is the beginning of the content. To remedy this, I can do further data cleaning to add newline characters between the document heading and content. Further text normalization in the post-processing step can remove any extra prefixes or suffixes from entities (e.g.: "Minnesota's" or "Ex-Itasca County").

#### False positives

The false positive rate for locations was high. Many of the false positives were laws or organizations that *contained* a location name. Examples are "Minnesota Board", "Minnesota Rules", "Minnesota Statutes", and "Minneapolis police". Perhaps the model was trained to tag any entity that *contains* a location. In that case, it may be impossible to solve the problem with post-processing.

Additionally, NER tags some entities that are entirely unrelated to a location. Government agencies or organizations such as "Transportation Security Administration", "Department of Homeland Security", and "Policy & Government Oversight Committee" were tagged as locations with fairly high confidence. This might also be because the model was trained on a broader definition of "locations" than we are using. Perhaps the individual(s) who trained the model annotated government agencies as locations because that was important for their domain and research problem. For *our* domain, we only want to tag geolocations. These other entities are false positives from the perspective of *our* research problem, but maybe not from *theirs*. 

The model is very accurate at detecting *valid* locations, and it was fine-tuned to detect locations. So, I believe this is not a problem or inaccuracy in the *model*, but instead a difference between their research problem and ours.

However, the model tagged "Duluth News" (a newspaper) and "George Floyd" as locations, but with low confidence. I believe this is a true error with the model because these entities are not locations in any context. 

#### Confidence Scores

The confidence scores were often misleading. The model sometimes assigned a low confidence score to valid location entities, and a high confidence score to invalid ones. In one instance, the locations "Ramsey County" and "Minnesota" had confidence scores around 50%, even though they are clearly locations. Therefore, we cannot simply set a confidence threshold to get rid of invalid locations. 

#### Final Recommendation

I believe the NER system for tagging locations is not yet accurate enough to be integrated into the pipeline. More refinements and improvements should be made before we can trust NER to tag locations reliably. More pre-processing of the documents and post-processing of the entities could significantly improve model accuracy. However, cleaning and parsing text can be time-consuming, so it may be worthwhile to instead investigate other location extraction models. 

A future direction could be to compare other location extraction models with `Inquirer_ner_loc`. `Inquirer_ner_loc` has exceptional accuracy for detecting locations, when compared to general NER models that detect multiple entity types. However, I did not yet compare it with other models that are trained specifically to extract *locations*. Perhaps other location extraction models do not exhibit the problems I described, and using them would save us the extensive work of pre- and post-processing. 

### Computational Time

It took a long time to run the NER system on multiple texts. On my "test set" of 20 texts, NER took 47.2 seconds to tag the dates in all the texts, with chunks of 100 words and no confidence threshold. This is not scalable to a real-world dataset. If we run NER on a full-size dataset (e.g.: 1,000-10,000 texts), it may take several hours.

Additionally, NER took much more time to tag longer documents than shorter ones. For the sake of obtaining results quickly, the test set included only documents that were less than 3,000 characters long. This is also a concern for real-world datasets because police documents include court cases, which are often long. 

Chunking by sentences takes more computational time than chunking every `n` words. This is because detecting sentence boundaries requires a tokenizer, whereas splitting it by words only requires simple Python string manipulation. Further investigation to determine the best chunking strategy should be done, particularly for locations. However, based on a brief comparison using a few documents, no chunking strategy consistently performs better than all the others. So, for now, I recommend chunking by every `n` words instead of chunking by sentences, to reduce computational time while maintaining NER accuracy. 

Parallel programming would be an effective next step to improve computational time. Additionally, instead of using `pandas` to store and manipulate NER results, I can use `numpy` or `polars` for faster performance. 

## Limitations and Known Bugs

- The NER system for dates tags article publication dates. This is technically a false positive because it's not the date of the incident. However, the LLM can identify that that's not the police incident date.
- NER will not detect dates where the day is an ordinal number (e.g.: "1st", "2nd", or "8th"). If that behavior is undesirable, the `validate_parsing()` function must be changed to allow dates in that format.
- Because the documents are USA-based, all dates are parsed with the month first instead of the day first. (e.g.: "10/12/2025" is read as "October 10th, 2025", not "December 12th, 2025"). 

## Discussion & Conclusion

Based on my findings so far, I believe NER is very promising for extracting dates. Combining NER with post-processing achieved nearly perfect accuracy for extracting dates. This was achieved using only a general NER model, and we did not need one that was fine-tuned to extract dates. If my findings hold true when the model is tested more rigorously, then I believe the NER system to extract dates, could almost completely mitigate the problem of LLM hallucinations. 

For our dataset, some entity types are more difficult to extract than others. In the context of our dataset, dates are simpler to extract. They come in far fewer formats than locations. It is also easier to identify where a date starts and ends in a sentence. Perhaps these are the reasons why NER performs so well for dates. 

Post-processing plays a crucial role in the accuracy of this NER system. Without it, there would be many false positives. If I hadn't used Pandas' built-in date parsing capabilities, then time durations, people's ages, incomplete dates, and other "extra" entities would be tagged as dates. That would result in a longer candidate list for the LLM, making the LLM's task more difficult. 

Police misconduct is a topic that concerns people's safety and lives. If we are ever to apply data science to this research area, it is our responsibility to create *highly, highly* accurate machine learning models. Correct information extraction is a necessity. Combining multiple models (NER and LLM) increases the accuracy, reliability, and robustness of our information extraction system. Therefore, I believe combining multiple models is a worthwhile direction for NLP research in the domain of police misconduct.

However, we must think very carefully about the capabilities, strengths, weaknesses, accuracy, and appropriateness of any NLP model we consider. For example, our overall goal is to extract important information about police incidents. Dr. Mendible was interested in information about who the victim was, the amount of force used, and the disciplinary action taken. These are complex NLP tasks that go beyond simple named entity recognition. NER has a limited scope of information it can extract. It can detect a limited, generic set of categories (e.g.: name, organization, location, date, ...). Answering more complex questions about an article, while taking the whole article into context, is where an LLM excels. 

I hope that NLP research (when used appropriately) will someday aid in police reform. Reading a large number of police incident documents takes far longer than using an NLP system to extract the information you seek. Using NLP systems like ours, one can quickly obtain concise information about specific police incidents from vast amounts of documents. Therefore, I hope that NLP systems enable activist organizations and local governments to take quicker beneficial action toward police reform, thus protecting the physical safety and lives of community members.

# Usage Instructions

## File descriptions
- `scraped_police_reports14.csv`: contains the raw data that Jesse Loi scraped from the CUAPB archive.
- `police_reports.csv`: contains the cleaned texts and a unique integer ID for each text
- `police_report_metadata.csv`: contains the following metadata about each police report text
    - `name`: the name and badge number of the officer mentioned in that text
    - `department`: the police department where that officer works
    - `url`: the URL that the text was scraped from
    - `text_id`: the unique integer ID for the text
- `ner.py`: contains functions to run NER on a dataset of documents
- `test_ner.ipynb`: contains the results of running NER on a small number of police documents
- `data_prep.py`: performs data cleaning and preparation on the raw data

## Prerequisites
- Install Python and Visual Studio Code on your local machine

## How to use this code

1. Clone this repository and open it in Visual Studio Code.
2. In VSCode, create a `venv` virtual environment and activate it.
3. Inside the virtual environment, run the following command to install the necessary packages: `pip install flair pandas numpy nltk ftfy`.
4. To run NER on our police incident documents:
- Read `police_reports.csv` into a Pandas DataFrame. (This is done in the "Load Data" section of `test_ner.ipynb`)
- Run the function `run_ner_on_dataframe()` in `ner.py` on that DataFrame. (For documentation on how to use that function, please refer to `ner.py`). Choose the following when calling `run_ner_on_dataframe()`:
    - The chunking strategy: by sentences or every `n` words
    - The value of `n`, if chunking by every `n` words
    - Optionally, a confidence threshold for the NER model
    - Whether to keep generic locations or only return specific ones
- NER may take a long time to finish running, especially when chunking by sentences, using a large DataFrame of documents, or including long documents in your DataFrame
- Your NER results will be in the DataFrame returned by `run_ner_on_dataframe()`.