import re

"""
Regular expressions, often referred to as regex, are powerful tools for pattern matching and text 
manipulation. They provide a concise and flexible way to describe and match patterns within strings. 
Regular expressions are widely used in various programming languages and are particularly useful for 
tasks such as data validation, searching, and text processing.

Regular expressions offer a versatile and efficient way to handle textual data. They enable programmers 
to define complex search patterns, making it easier to locate, extract, or manipulate specific information 
within strings. Whether you're validating user input, searching for specific patterns in a document, or 
extracting data from a large dataset, regular expressions can significantly streamline these tasks.
"""

print("*********** hey, this is just the test !!!! **********")


# This function demonstrates the basic usage of RegEx in python.
# DO NOT edit this function
def regExFunctions():
    # Text to apply regular expression to
    text = "In this forest, there are many animals. This morning, I already saw five dogs, two cats, and one coyote. The dogs were frolicking in the meadow."

    # Define a pattern
    # In this case we are matching all occurences of "dogs"
    # raw strings (r before string) are common for regex expressions: they don't handle escaped characters
    pattern = r"dogs"

    # re.search() methods return the first occurence of the match
    match = re.search(pattern, text)
    print('re.search: ', match)

    # Use re.findall() to find all matches in a given string
    all_matches = re.findall(pattern, text)

    print('re.findall: ', all_matches)

    # Use re.sub() to replace matched patterns
    replacement = r"foxes"
    modified_text = re.sub(pattern, replacement, text)
    print('re.sub: ', modified_text)


def regExExerciseOne(input_text):
    """
    TODO: Complete this function so that it finds and replace all "has" with "owns"

    Example: Alice has 3 apples, Bob has 5 oranges, and Charlie has 2 bananas. David has 8 grapes,
            and Emily has 4 peaches.

    Output: Alice owns 3 apples, Bob owns 5 oranges, and Charlie owns 2 bananas. David owns 8 grapes,
            and Emily owns 4 peaches."
    """
    pattern = r"has"
    replacement = r"owns"
    modified_text = re.sub(pattern, replacement, input_text) #"Alice has 3 apples, Bob has 5 oranges, and Charlie has 2 bananas. David has 8 grapes, and Emily has 4 peaches."

    return modified_text

"""
Character Classes:
You can match multiple characters by listing them inside the hard bracket '[ ]'. 
The following pattern will match characters 'a', 'e', 'i', 'o', and 'u'

pattern = r"[aeiou]"
text = "hello, world!"

matches = re.findall(pattern, text)

print(matches) # ['e', 'o', 'o', 'e', 'o']

pattern = r"[A-Z]" #Matches all uppercase alphabet
pattern = r"[A-Za-z0-9]" #Matches all alphanumeric

Some commonly used character classes have shorthands. 
- `\d`== `[0-9]`
- `\w` == `[A-Za-z]`
- `\s` == whitespaces
"""


def regExExerciseTwo(input_text):
    # TODO: Complete this function so that it finds all numbers and replaces them with the letter 'X'
    pattern = r"[0-9]"
    replacement = r"X"
    modified_text = re.sub(pattern,replacement,input_text)

    return modified_text


"""
Quantifiers allows us to specify number of occurrences of a character or group in a given pattern

Matches exactly 4 characters of lower case alphabet
pattern = r"[a-z]{4}"

Matches digits in patterns of 333-333-3333
pattern = r"\d{3}-\d{3}-\d{4}"

Matches 0 or more 'a's
pattern = r"a*"

Matches 1 or more a's
pattern = r"a+"

Matches 2 or more a's
pattern = r"a{2,}"
"""


def regExExerciseThree(input_string):
    """
    TODO: This time we will build a simple email validator. Complete the function so that it returns
          True if the input string is of the following pattern: string@string.xxx, that is, one or more
          string followed by "@" symbol, followed by 1 or more string again, followed by ".", then
          followed by a one to three character string. All strings should be lowercase, uppercase, and/or
          numeric characters only

    Example:
    example@example.com
    Outcome:
    True

    Example2:A
    aweih@eiwhwoih.
    Outcome:
    False, as it doesn't contain characters following "." symbol.
    """
    pattern = r"[a-zA-Z0-9]+@[a-zA-Z0-9]+\.[a-zA-Z]{1,3}"
    match = re.search(pattern, input_string)
    if match:
        flag = True
    else:
        flag = False
    return flag