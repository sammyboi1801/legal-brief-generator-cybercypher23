from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import streamlit as st

tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
model_large = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
model_small = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
classifier = pipeline("zero-shot-classification",model="facebook/bart-large-mnli")
question_answerer = pipeline("question-answering", model='distilbert-base-cased-distilled-squad')
model = AutoModelForSeq2SeqLM.from_pretrained("nsi319/legal-pegasus")


def create_arguments(text1):
    #text = """What is the conclusion according to this text?: On March 5, 2021, the Securities and Exchange Commission charged AT&T, Inc. with repeatedly violating Regulation FD, and three of its Investor Relations executives with aiding and abetting AT&T's violations, by selectively disclosing material nonpublic information to research analysts. According to the SEC's complaint, AT&T learned in March 2016 that a steeper-than-expected decline in its first quarter smartphone sales would cause AT&T's revenue to fall short of analysts' estimates for the quarter. The complaint alleges that to avoid falling short of the consensus revenue estimate for the third consecutive quarter, AT&T Investor Relations executives Christopher Womack, Michael Black, and Kent Evans made private, one-on-one phone calls to analysts at approximately 20 separate firms. On these calls, the AT&T executives allegedly disclosed AT&T's internal smartphone sales data and the impact of that data on internal revenue metrics, despite the fact that internal documents specifically informed Investor Relations personnel that AT&T's revenue and sales of smartphones were types of information generally considered "material" to AT&T investors, and therefore prohibited from selective disclosure under Regulation FD. The complaint further alleges that as a result of what they were told on these calls, the analysts substantially reduced their revenue forecasts, leading to the overall consensus revenue estimate falling to just below the level that AT&T ultimately reported to the public on April 26, 2016. The SEC's complaint, filed in federal district court in Manhattan, charges AT&T with violations of the disclosure provisions of Section 13(a) of the Securities Exchange Act of 1934 and Regulation FD thereunder, and charges Womack, Evans and Black with aiding and abetting these violations. The complaint seeks permanent injunctive relief and civil monetary penalties against each defendant. The SEC's investigation was conducted by George N. Stepaniuk, Thomas Peirce, and David Zetlin-Jones of the SEC's New York Regional Office. The SEC's litigation will be conducted by Alexander M. Vasilescu, Victor Suthammanont, and Mr. Zetlin-Jones. The case is being supervised by Sanjay Wadhwa."""
    text = "write a legal brief for: "+text1
    input_tokenized = tokenizer.encode(text, return_tensors='pt',max_length=1024,truncation=True)
    summary_ids_large = model_large.generate(input_tokenized,
                                    num_beams=9,
                                    no_repeat_ngram_size=3,
                                    length_penalty=2.0,
                                    min_length=150,
                                    max_length=600,
                                    early_stopping=True)

    summary_argument = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summary_ids_large][0]

    return summary_argument

def conclusion_text(text1):
    text = "Who is guilty in this case legally?: " + text1
    input_tokenized = tokenizer.encode(text, return_tensors='pt', max_length=1024, truncation=True)
    summary_ids_large = model_small.generate(input_tokenized,
                                             num_beams=9,
                                             no_repeat_ngram_size=3,
                                             length_penalty=2.0,
                                             min_length=30,
                                             max_length=150,
                                             early_stopping=True)

    summary_argument = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summary_ids_large][0]

    return summary_argument

def zero_shot_classifier(text):
    sequence_to_classify = text
    labels = ['Arson','Organized Crime','Assault','Kidnapping','Bribery','Property Crime','Theft','Identity Theft','Motor Vehicle Theft','Felony','Domestic Violence','Human Trefficking','Extortion','Burglary','Manslaughter']
    print(classifier(sequence_to_classify, labels))


def question_answerer(question,text):
    context = text

    result = question_answerer(question=question, context=context)

    return f"Answer: '{result['answer']}', score: {round(result['score'], 4)}, start: {result['start']}, end: {result['end']}"


def facts_retriever(text):

    input_tokenized = tokenizer.encode(text, return_tensors='pt', max_length=1024, truncation=True)
    summary_ids = model.generate(input_tokenized,
                                 num_beams=9,
                                 no_repeat_ngram_size=3,
                                 length_penalty=2.0,
                                 min_length=150,
                                 max_length=250,
                                 early_stopping=True)
    summary = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summary_ids][0]

# def involve_text(text1):
#     text = "Who is guilty in this case legally?: " + text1
#     input_tokenized = tokenizer.encode(text, return_tensors='pt', max_length=1024, truncation=True)
#     summary_ids_large = model_small.generate(input_tokenized,
#                                              num_beams=9,
#                                              no_repeat_ngram_size=3,
#                                              length_penalty=2.0,
#                                              min_length=30,
#                                              max_length=150,
#                                              early_stopping=True)
#
#     summary_argument = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summary_ids_large][0]
#
#     return summary_argument


st.write("""
# Lawyer AI :closed_book:
"""
)

text = st.text_input("Enter the facts")

if text!="":
    arguments = create_arguments(text)

    st.subheader("Arguments")
    st.markdown(arguments)

    conclusion = conclusion_text(text)

    st.subheader("Conclusion :exclamation:")
    st.markdown(conclusion)

    with st.sidebar:
        st.subheader("Category")
        zero_shot_classifier(text)

        question = st.text_input("")
        question_answerer(question,text)

    # verdict=involve_text(text)
    # st.subheader("Verdict")
    # st.markdown(verdict)

else:
    st.write("Enter text...")


