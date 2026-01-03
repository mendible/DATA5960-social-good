""" This file contains functions to perform NER on the police documents. """
from nltk.tokenize import sent_tokenize
import pandas as pd
from flair.data import Sentence
import numpy as np
import string
from IPython.display import display
import re

# Dict where keys are all possible strings that represent a valid month, and values are the non zero-padded integer corresponding to that month
VALID_MONTHS = {
    "jan": 1, "january": 1,
    "feb": 2, "february": 2,
    "mar": 3, "march": 3,
    "apr": 4, "april": 4,
    "may": 5,
    "jun": 6, "june": 6,
    "jul": 7, "july": 7,
    "aug": 8, "august": 8,
    "sep": 9, "sept": 9, "september": 9,
    "oct": 10, "october": 10,
    "nov": 11, "november": 11,
    "dec": 12, "december": 12,
}

def tokenize_text(text: str) -> list:
    """ 
    Split a text string into alphabetical or numerical tokens. 
    These tokens should be the day, month, and year components.
    Examples:
        - tokenize_text("Feb. 9, 2022") -> ["feb", "9", "2022"]
        - tokenize_text("2020/08/09") -> ["2020", "08", "09"]

    Params
        - text: the original string to split
    Returns: list of all alphabetical or numerical tokens in the text. 
             (Note: Uppercase letters in the original text are converted to lowercase in the returned list).
    """

    # all groups of consecutive letters or consuecutive digits in the text.
    matches = re.findall(pattern = r"[A-Za-z]+|\d+", string = text.lower())

    matches = [match for match in matches if len(match) > 0] # Remove empty strings from the matches
    return matches

def validate_parsing(row: np.ndarray) -> bool:
    """ 
    Checks if a date entity in an article was parsed correctly by Pandas. 
    Returns True if the parsed date object is the same date as the original text, and False otherwise.
    (If there's a missing month or day in the original text, Pandas may set it to a default value in the parsed date. That is incorrect parsing).

    Params:
        - row: a single row of the `combined_df` dataframe defined in `post_process_dates()`. 
               MUST be a NumPy array of shape (2,).
               The first array element is the original text, with type `str`. The second element is the parsed date, which is either a `datetime.date` object or `NaT`. 
               (If the array shape or types are incorrect, this function will not behave as expected).

    Returns: boolean for whether the parsed date is correct or not. Returns False if the date is `NaT`
    """
    
    text, parsed_date = row  # unpack the row into the text string and the parsed date, separately

    # If the parsed date is `NaT`, return False
    if pd.isna(parsed_date):
        return False

    components = tokenize_text(text) # split text into a list of alphanumeric components (should be day, month, and year)

    # If the text doesn't have explicit month, day, and year components, then Pandas may fill in a default value for the missing component.
    # So, if the text has less than 3 components and Pandas still parsed the text, then the parsing is incorrect.
    if len(components) < 3:
        return False

    year_4_digit = str(parsed_date.year)           # the 4-digit year of the parsed date (e.g.: "2001")
    year_2_digit = f"{parsed_date.year % 100:02d}" # the 2-digit year of the parsed date (e.g.: "01")
    is_year_valid = year_4_digit in components or year_2_digit in components # The parsed year is correct if the 2-digit parsed year OR the 4-digit parsed year is in the original text. Check this.
 
    # Check if the day in the parsed date equals the day in the component string.
    # `f"{parsed_date.day:02d}"` converts the day integer to 2 digits by zero-padding (if needed).
    # The day must be an integer token in the text, not an alphabetical token.
    # NOTE: Does not allow days written as "8th", "1st", etc. Months can be a combination of alphabetic & numeric characters - e.g.: 8th, 3rd, 1st, ... MAYBE handle this case? Or document it.
    is_day_valid = str(parsed_date.day) in components or f"{parsed_date.day:02d}" in components

    # Check if month is valid
    is_month_valid = (
        # Checks if the integer month in the parsed date is a component of the text, 
        # or if the zero-padded integer month in the parsed date is a component.
        str(parsed_date.month) in components or f"{parsed_date.month:02d}" in components

        # For textual months: check if any of the components is equal to a valid textual month.
        # `any()` returns True if any of the items in the list are True. If this is True, we have a valid textual month.
        or any([VALID_MONTHS.get(comp) == parsed_date.month for comp in components])
    )

    # The parsing is valid only if the year, month, and day are all valid
    return is_month_valid and is_day_valid and is_year_valid

def split_text(text: str, method: str, n: int = -1) -> list[Sentence]:
    """ 
    Split a string of text into chunks using the specified method.

    Parameters:
        - `text`: the text to split

        - `method`: How to split the text. To split by sentences, pass `method = "sentences"`. 
                  To split the text every `n` words, pass `method = "every_n_words"`, and pass the number of words for `n`. 
        - `n`: if you're splitting the text by words, this is the number of words in each chunk of text.

    Returns: a list of Flair sentence objects, in order, each one representing a chunk
    """

    sentences = [] # list of Flair Sentence objects, one for each chunk of text.
    
    # Split every n words
    if method == "every_n_words":
        words = text.split() # split the text into a list of words by splitting it on every whitespace
        chunks = [' '.join(words[i:i + n]) for i in range(0, len(words), n)] # split text into a list of chunks of n words
        sentences = [Sentence(chunk) for chunk in chunks] # make each chunk into a Flair Sentence object

    # Split by sentences using a sentence tokenizer
    else:
        sentence_texts = sent_tokenize(text) # use a tokenizer to get a list of sentences, as strings
        sentences = [Sentence(text) for text in sentence_texts] # make each sentence string into a Flair Sentence object

    # Link the sentences so that each Sentence object has a pointer to the previous sentence and the next sentence, to preserve context information.
    Sentence.set_context_for_sentences(sentences) 
    return sentences


def print_text(text):
    """
    Print an article text in a more readable way. 
    """

    flair_sentences = split_text(text, method="every_n_words", n=15) # Flair Sentence objects for each 15-word chunk
    text_chunks = [sent.text for sent in flair_sentences] # convert those sentences to strings
    clean_text = "\n".join(text_chunks) # Make a string with each sentence on its own line
    print(clean_text)


def post_process_locations(locations_df: pd.DataFrame, keep_generic_locs: bool) -> list[str]:
    """ 
    Post-process the predicted locations by removing punctuation and whitespace.

    Parameters:
        - locations_df: DataFrame with columns for the locations predicted by NER, and the confidence scores. The column for the locations MUST be named "entities"
        - keep_generic_locs: whether to keep generic locations like "city", "at the bridge", "hospital", "sidewalk", "a street", etc.
          This is as opposed to specific locations, such as "Hennepin County", "Main Street", "704 West Street, Minneapolis, MN", "Minneapolis", etc.
    
    Returns: the post-processed DataFrame of locations and scores.
    """    

    locations_df["entities"] = locations_df["entities"].astype("str") # convert the column to string, if it's not already
    
    # Remove any punctuation and whitespace from the end of every string in `locations`.
    # `string.punctuation` and `string.whitespace` are constants containing all punctuation and whitespace characters.
    locations_df["entities"] = locations_df["entities"].str.strip(string.punctuation + string.whitespace)

    locations_df = locations_df.loc[locations_df["entities"].str.len() > 1] # Remove any strings that are empty or have length 1

    # Some of the detected locations were URLs, like `minnesota.cbslocal.com`. 
    # To match URLs like that one, we will look for locations that DON'T contain whitespace, DO contain an underscore or non-word character, AND are all lowercase.
    # Condition for whether a text is a URL:
    url_condition = (~locations_df["entities"].str.contains(r"\s+")) & (locations_df["entities"].str.contains(r"\W+|_+")) & (locations_df["entities"].str.islower())

    locations_df = locations_df.loc[~url_condition] # Keep only locations that DON'T match that condition

    # If we are not keeping generic locations, filter out all-lowercase strings, because 
    # specific street/city/state names start w/ capital letters. 
    # (NOTE: this isn't a very good or foolproof way to check, b/c generic locations can be uppercase too).
    if not keep_generic_locs:
        locations_df = locations_df.loc[~locations_df["entities"].str.islower()]

    return locations_df.reset_index(drop=True)

def post_process_dates(df: pd.DataFrame) -> pd.DataFrame:
    """ 
    Post-process the NER predictions for dates, keeping valid dates only. 

    Parameters:
            - df: DataFrame of predicted entities with the following columns:
                - `entities` (`str`): the dates predicted by NER 
                - `score` (`float`): confidence scores.
      
    Returns: DataFrame with the following columns:
                - `entities` (`datetime.date`): the entities represented as `date` objects, in the format "MM/DD/YYYY"
                - `scores`: same as above
             Rows where the entities cannot be parsed to `datetime.date` objects are removed.
    """

    df["entities"] = df["entities"].astype("str")                                       # convert the column to string, if it's not already
    df["entities"] = df["entities"].str.strip(string.punctuation + string.whitespace)   # Remove any punctuation and whitespace from the end of every string
    df = df.loc[df["entities"].str.len() > 1].copy()                                           # Drop rows with empty or length-1 strings 

    """
    Parse each text string to a `datetime.date` object. If it can't be parsed, set to `NaT`. Add the parsed dates to `df`.
        - `dayfirst = False`: parse dates in MM/DD/YYYY format instead of DD/MM/YYYY 
        - `format = 'mixed'`: infer the date format for each text individually (i.e.: texts are not required to be in a certain format). 
    """
    df["dates"] = pd.to_datetime(df["entities"], format="mixed", dayfirst=False, errors="coerce").dt.date

    df = df.loc[df["dates"].notna()]                                                   # Keep only the rows where Pandas could parse the date.     
    is_valid = df[["entities", "dates"]].apply(validate_parsing, axis = 1, raw = True) # Check if parsing is correct. Returns a boolean Series for whether each text string was parsed correctly.

    # Keep only rows where the date parsing is valid. Change the `entities` column to contain the parsed date objects instead of the text strings.
    final_df = df.loc[is_valid].drop(columns="entities").rename(columns={"dates": "entities"}).reset_index(drop=True)
    return final_df

def predict_entities(text: str, text_id: int, model, conf_thresh: float|None, entity_tag: str|None, splitting_method: str|None, 
                                  keep_generic_locs: bool = False, n: int = -1) -> pd.DataFrame:
    """ 
    Params
        - text: the text to run NER on
        - text_id: the unique integer ID corresponding to this text (this ID is found in the dataframe of articles)
        - `model`: The Flair model to use for NER
        - `conf_thresh`: threshold for the confidence score. Only entities with scores above this threshold will be kept.
                       MUST be in the open interval (0, 1). To keep all entities, pass `None`.
        - `entity_tag`: the Flair tag of the entity type you want to predict. For example, if you're predicting dates using
                     `flair/ner-english-ontonotes-large`, pass "DATE". This MUST be the exact tag that the model uses.
                     You can find these in the model documentation on HuggingFace. 
                     To predict locations, pass `None`, which returns all entities.
        - `method`: How to split the text. To split by sentences, pass `method = "sentences"`. 
                  To split every `n` words, pass `method = "every_n_words"`, and pass the number of words for `n`. 
                  If you don't want to split the text and want the model to run the prediction on the whole test as one chunk, pass `None`. 
        - `n`: if you're splitting the text by words, this is the number of words in each chunk of text.
        - `keep_generic_locs`: if you're predicting locations, this parameter determines whether to keep generic locations like "city", "at the bridge", "hospital", "sidewalk", etc.
          This is as opposed to specific locations, such as "Hennepin County", "Main Street", "704 West Street, Minneapolis, MN", "Minneapolis", etc.
          If you're predicting dates, this parameter is not used.

    Returns: a DataFrame containing the following columns
        - `entities`: all the predicted entities
        - `score`: confidence score for each entity
        - `text_id`: The unique ID of the text that each entity is from. (This is just the ID of the text that was passed in).
    """
    
    # Get the text or list of chunks that will be passed to the model for prediction.
    # If a splitting method is specified, get the splitted text as a list of Flair Sentence objects. 
    # Otherwise, make one Sentence object for the whole text.
    text_chunks = split_text(text = text, method = splitting_method, n = n) if splitting_method is not None else Sentence(text)

    model.predict(text_chunks)

    # Dataframe for NER results. Add a blank row because the behavior of `pd.concat()` might be deprecated when using it with
    # empty dataframes.
    ner_df = pd.DataFrame.from_dict({"label": [None], "entities": [None], "score": [np.nan]})

    # If we did not split the text, then turn the text into a one-element list.
    if not isinstance(text_chunks, list):
        text_chunks = [text_chunks]

    # Loop through each chunk and add NER results to the big dataframe
    for chunk in text_chunks:
            label_objects = chunk.get_labels() # Flair Label objects for all the predicted entities in this chunk

            # If there are any NER entities predicted, add them to the dataframe
            if len(label_objects) > 0:
                labels = [label_obj.value for label_obj in label_objects]             # all entity tags in this chunk
                entities = [label_obj.data_point.text for label_obj in label_objects] # all predicted entities in this chunk
                scores = [label_obj.score for label_obj in label_objects]             # confidence scores of all entities

                df = pd.DataFrame.from_dict({"label": labels, "entities": entities, "score": scores}) # dataframe of all entities and scores for this chunk
                ner_df = pd.concat([ner_df, df], axis = 0) # add predictions for this chunk to the big dataframe

    # If `entity_tag` is specified, keep only entities w/ that tag.
    if entity_tag is not None:
        ner_df = ner_df.loc[ner_df["label"] == entity_tag]

    # Else, drop the null row, because if we don't filter by the entity tag, it doesn't get filtered out.
    elif not ner_df.empty:
        ner_df = ner_df.reset_index(drop=True).drop(index=0) 

    ner_df = ner_df.drop(columns=["label"]) # drop label column, don't need it

    # apply confidence threshold if one is specified
    if conf_thresh is not None:
        ner_df = ner_df.loc[ner_df["score"] >= conf_thresh]

    ner_df = ner_df.reset_index(drop=True)

    # Post-processing. If an entity tag is passed, that means we're predicting dates. If no entity tag is passed, we're predicting locations.
    ner_df = post_process_dates(ner_df) if entity_tag is not None else post_process_locations(ner_df, keep_generic_locs)
    ner_df = ner_df[["entities", "score"]]

    # have a column for the text id in the returned dataframe. This is the same in the entire dataframe, since we're running NER on the same text.
    text_id_col = np.full(shape = len(ner_df), fill_value=text_id) 
    ner_df["text_id"] = text_id_col

    return ner_df.to_numpy()

def run_ner_on_dataframe(model, text_df: str, conf_thresh: float | None, entity_tag: str | None, 
                                        splitting_method: str, 
                                        keep_generic_locs: bool = False, 
                                        n: int = -1
                                      ) -> pd.DataFrame:
    """
    Run NER on each row of a DataFrame of texts, extracting only a certain entity type (e.g.: location or date).
    
    Parameters:
        - `model`: The Flair model to use for NER
        - `text_df`: DataFrame containing article texts. The texts MUST be in a column called "text". There MUST be a column called `text_id` with the unique integer ID for each text.
           The DataFrame must not have any null or missing texts.
        - `conf_thresh`: threshold for confidence scores. Only entities with scores above this threshold will be kept.
                        MUST be in the open interval (0, 1). To keep all entities, pass `None`.
        - `entity_tag`: the tag of the entity type you want to predict. For example, if you're predicting dates using
                        `flair/ner-english-ontonotes-large`, pass "DATE". This MUST be the exact tag that the model uses.
                        You can find these in the model documentation on HuggingFace. 
                        If you pass `None`, all predicted entities will be returned
        - `method`: How to split the text. To split by sentences, pass `method = "sentences"`. 
                    To split every `n` words, pass `method = "every_n_words"`, and pass the number of words for `n`. 
                    If you don't want to split the text and want the model to run the prediction on the whole test as one chunk, pass `None`. 
        - `n`: if you're splitting the text by words, this is the number of words per text chunk.
        - `keep_generic_locs`: if you're predicting locations, this parameter determines whether to keep generic locations (e.g.: "city", "at the bridge", "hospital", "sidewalk", etc.).
            This is as opposed to specific locations, (e.g.: "Hennepin County", "Main Street", "704 West Street, Minneapolis, MN", "Minneapolis", etc.).
            If you're predicting dates, this parameter is not used.
        
    Returns: a DataFrame of NER entities for each text, containing the following columns: 
                - "text": text column of `text_df`
                - "text_id": `text_id` column of `text_df`
                - "entities": the predicted entities in the texts, with one entity per row.
                - "scores": confidence scores for those entities, with one score per row.
                - All other columns that were in `text_df`.
                - (Note: "text" and "text_id" (and any other columns in `text_df`) will be duplicated across rows, with one row per entity).
    """

    # check if the index is equal to the default integer index from 0 to n-1. If not, reset it to that index.
    if not text_df.index.equals(pd.RangeIndex(stop = len(text_df))):
        text_df = text_df.reset_index(drop=True)

    # Result of running NER on each text in the dataframe. A Series where each element is a 2D NumPy array
    # `apply()` applies the lambda function to each row of `text_df`, to run NER on each text.
    ner_result = text_df.apply(

        # This lambda function takes a row of `text_df` as an input, extracts the text and text ID from the row, and runs NER on the text.
        lambda row: predict_entities(
            text=row["text"],           # Extract the text column from the row
            text_id=row["text_id"],     # extract the text_id column from the row
            model=model,
            conf_thresh=conf_thresh,
            entity_tag=entity_tag,
            splitting_method=splitting_method,
            keep_generic_locs=keep_generic_locs,
            n=n,
        ),

        axis=1
    )

    """ Get the NER results into a more readable format. """
    result_as_numpy = ner_result.to_numpy() # NER result as numpy array
    result_as_list_restructured = [] # restructured list of results we'll make into a dataframe

    for i in range(len(result_as_numpy)):
        text_result = result_as_numpy[i] # all entities and scores for one text
        entities = text_result[:, 0].tolist()     # all entities for that text
        scores = text_result[:, 1].tolist()       # all scores for that text
        text_list = [entities, scores] # nested list of entities and scores for this text
        result_as_list_restructured.append(text_list) # add that to big list

    # a dataframe with one row for each text, where the "entities" column is a list of all entities for that row,
    # and the "score" column is the corresponding list of scores.
    ner_results_df = pd.DataFrame(result_as_list_restructured, columns=["entities", "scores"]) 

    combined_df = pd.concat([text_df, ner_results_df], axis = 1) # Add the columns with NER results to the original dataframe. 

    # Explode all list-like columns so that each list element is on its own row. The `text` column is duplicated for all exploded rows.
    final_df = combined_df.explode(["entities", "scores"]).sort_values(by=["text_id", "entities"]).reset_index(drop=True)

    return final_df

