Title: Back Translation for Text Augmentation with Google Sheets
Date: 2020-02-19 16:13
Modified: 2020-02-19 16:13
Category: nlp
Slug: back-translation-in-google-sheets
Summary: Learn how to augment existing labelled text data for free using google sheets.
Authors: Amit Chaudhary
Status: published


When working on Natural Language Processing applications such as text classification, collecting enough labelled examples for each category manually can be difficult. In this article, I will go over an interesting technique to augment your existing text data automatically using a technique called back translation.

## Introduction to Back Translation
The key idea of back translation is very simple. We create augmented version of a sentence using the following steps:

1. You take the original text written in English  
2. You convert it into another language (say French) using Google Translate  
3. You convert the translated text back into English using Google Translate   
4. Keep the augmented text if the original text and the back-translated text are different. 

![](/images/backtranslation-en-fr.png){.img-center}
<p class="has-text-centered has-text-grey">
Figure: Back Translation
</p>

## Using Back Translation in Google Sheets
We need a machine translation service to perform the translation to a different language and back to English. Google Translate is the most popular service for this purpose, but you need to get an API key to use it and it is a paid service. 

Luckily, Google provides a handy feature in their Google Sheets webapp, which we can leverage for our purpose.

### Step 1: Load your data
Let's assume we are building a sentiment analysis model and our dataset has sentence and their associated labels. We can load it into Google Sheets by importing the Excel/CSV file directly.
![](/images/backtranslation-sheets-step-1.png)

## Step 2: Add a column to hold augmented data
Add a new column and use the `GOOGLETRANSLATE()` function to translate from English to French and back to English.
![](/images/backtranslation-sheets-step-2.png)

The command to place in the column is
```js
=GOOGLETRANSLATE(GOOGLETRANSLATE(A2, "en", "fr"), "fr", "en")
```
Once the command is placed, press Enter and you will see the translation.
![](/images/backtranslation-sheets-step-2.2.png)

Now, select the first cell of "Backtranslated" column and drag the small square at the bottom right side below to apply this formula over the whole column 
![](/images/backtranslation-sheets-step-2.3.png)

This should apply to all your training texts and you will get back the augmented version.
![](/images/backtranslation-sheets-step-2.4.png)

## Step 3: Filter out duplicated data
For texts where the original text and what get back from `back translation` is same, we can filter them out programmatically by comparing the original text column and the augmented column. Then, only keep responses that have `True` value in the `Changed` column.
![](/images/backtranslation-sheets-step-3.2.png)

## Step 4: Export your data
You can download your data as a CSV file and augment your existing training data.

## Example Sheet
Here is a [Google Sheet](https://docs.google.com/spreadsheets/d/1pE9RAukrc4S9jf22RxVr_vEBqN9_DyZaRY8QQRek8Fs/edit#gid=2000059744) demonstrating all the four steps above. You can refer to that and make a copy of it to test things out.

## Conclusion
Back translation offers an interesting approach when you've small training data but want to improve performance of your model.