feedback_prompt = """
                    You got the following feedback to the task you did. Here is the feedback:
                    
                    Feedback text:
                    {response_text}
                    
                    Restructure this text and give the text in a clear tone. 
                    No need for any polite and formal phrases.
                """