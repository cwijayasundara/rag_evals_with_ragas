test_call_transcript_1 = """call_id: 271354 call_date_time: 08:07:2023 14:22:05 call_duration: 00:05:32 agent: Lucas 
Green call_transcript: agent(Lucas Green) How can I be of service today? customer(Sarah Bennett) I'm having trouble 
with a payment not going through. agent(Lucas Green) Let's resolve that for you, Sarah. Can I have the last four 
digits of your SSN for security purposes? customer(Sarah Bennett) Sure, they are 4321. agent(Lucas Green) Great, 
we'll check on that payment now."""

test_call_transcript_2 = """call_id: 872394 call_date_time: 15:07:2023 11:05:32 call_duration: 00:03:45 agent: 
Samantha Lewis call_transcript: agent(Samantha Lewis) Thank you for calling, how can I assist you? customer(Harry 
Green) Hello, I found a weird charge on my card statement. agent(Samantha Lewis) I understand your concern. May I 
have the email address linked to your account for verification? customer(Harry Green) Sure, 
it's harry.green@email.com. agent(Samantha Lewis) And for additional security, could you provide the last 4 digits of 
your SSN? customer(Harry Green) Yes, it's 5521. agent(Samantha Lewis) Thank you, Harry. Let's look into this 
transaction; what's the date and amount? customer(Harry Green) The suspicious charge was on the 14th, 
for $250. agent(Samantha Lewis) Okay, I will investigate this matter and get back to you shortly. Is there anything 
else you need help with? customer(Harry Green) No, that's all, thanks. agent(Samantha Lewis) You're welcome. Have a 
good day.
"""

test_call_transcript_query_1 = """List all the calls where agent Lucas Green was involved"""
test_call_transcript_query_2 = """list all the calls happened on the 15th of July 2023"""
