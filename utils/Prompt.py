import json # [核心修改] 添加这一行

EXP_ALL_USERS_TEST = ["persona_36"]

EXP_ALL_USERS = ["persona_288", "persona_45", "persona_36", "persona_13", "persona_80", "persona_14", "persona_112", "persona_117", "persona_16", "persona_280", "persona_195", 
    "persona_283", "persona_58", "persona_48", "persona_296", "persona_64", "persona_113", "persona_82"]

Roles = {
    'CB':['Buyer', 'Seller', 'BuyerCritic', 'SellerCritic'],
    'P4G':['Persuader', 'Persuadee', 'PersuaderCritic', 'PersuadeeCritic']
}

CB_Critic_Prompt = """
    Given a conversation between a Buyer and a Seller, please decide whether the Buyer and the Seller have reached a deal at the end of the conversation.
"""


Buyer_First_Sentence = """Hi, how much is the {item_name}?"""

Persuader_First_Sentence = """Hello! How are you?"""

human_bargain_strategy_instruction_map = {
    "Greeting": "Please say hello or chat randomly.",
    "Ask a question": "Please ask any question about product, year, price, usage, etc.",
    "Answer a question": "Please provide information about the product, year, usage, etc.",
    "Propose the first price": "Please initiate a price or a price range for the product.",
    "Propose a counter price": "Please propose a new price or a new price range.",
    "Use comparatives": "Please propose a vague price by using comparatives with existing price.",
    "Confirm information": "Please ask a question about the information to be confirmed.",
    "Affirm confirmation": "Please give an affirmative response to a confirm.",
    "Deny confirmation": "Please give a negative response to a confirm.",
    "Agree with the proposal": "Please agree with the proposed price.",
    "Disagree with a proposal": "Please disagree with the proposed price.",
}

CBAct = {
    'greet': 'Please say hello or chat randomly.',
    'inquire': 'Please ask any question about product, year, price, usage, etc.',
    'inform': 'Please provide information about the product, year, usage, etc.',
    'propose': 'Please initiate a price or a price range for the product.',
    'counter': 'Please propose a new price or a new price range.',
    'counter-noprice': 'Please propose a vague price by using comparatives with existing price.',
    'confirm': 'Please ask a question about the information to be confirmed.',
    'affirm': 'Please give an affirmative response to a confirm.',
    'deny': 'Please give a negative response to a confirm.',
    'agree': 'Please agree with the proposed price.',
    'disagree': 'Please disagree with the proposed price.'
}

# 应用到P4G任务中
wrong_bargain2persuader_strategy_instruction_map = {
    "Greeting": "Please say hello or chat randomly.",
    "Ask a question": "Please ask any question about the cause, its purpose, its impact, its legitimacy, etc.",
    "Answer a question": "Please provide information about the cause, its purpose, its impact, its legitimacy, etc.",
    "Propose the first price": "Please suggest an initial amount of money to be donated.",
    "Propose a counter price": "Please suggest a new amount or a new range for a donation, perhaps arguing for a different level.",
    "Use comparatives": "Please suggest a vague amount by using comparatives, like 'more than this' or 'less than that'.",
    "Confirm information": "Please ask a question to clarify information about the cause, its needs, or the requested donation.",
    "Affirm confirmation": "Please give an affirmative response to a confirm.",
    "Deny confirmation": "Please give a negative response to a confirm.",
    "Agree with the proposal": "Please agree to donate the proposed amount.",
    "Disagree with a proposal": "Please refuse to donate the proposed amount (or propose a different solution).",
}

# "Foot in the Door": "Please use the strategy of starting with small donation requests to facilitate compliance followed by larger requests.",
human_persuader_strategy_instruction_map = {
    "Greeting": "Please say hello or chat randomly." ,
    "Logical Appeal": "Please use reasoning and evidence to convince the persuadee.",
    "Emotion Appeal": "Please elicit the specific emotions to influence the persuadee.",
    "Credibility Appeal": "Please use credentials and cite organizational impacts to establish credibility and earn the user’s trust. The information usually comes from an objective source (e.g., the organization’s website or other well-established websites).",
    "Self Modeling": "Please use the self-modeling strategy where you first indicate the persuadee's own intention to donate and choose to act as a role model for the persuadee to follow.",
    "Foot in the Door": "Please use the strategy of starting with small donation requests to facilitate compliance followed by larger requests.",
    "Personal Story": "Please use narrative exemplars to illustrate someone's donation experiences or the beneficiaries' positive outcomes, which can motivate others to follow the actions.",
    "Donation Information": "Please provide specific information about the donation task, such as the donation procedure, donation range, etc. By providing detailed action guidance, this strategy can enhance the persuadee’s self-efficacy and facilitate behavior compliance.",
    "Source Related Inquiry": "Please ask if the persuadee is aware of the organization (i.e., the source in our specific donation task).",
    "Task Related Inquiry": "Please ask about the persuadee's opinion and expectation related to the task, such as their interests in knowing more about the organization.",
    "Personal Related Inquiry": "Please ask about the persuadee's previous personal experiences relevant to charity donation.",
}

wrong_persuader2bargain_strategy_instruction_map = {
    "Greeting": "Please say hello or chat randomly." ,
    "Logical Appeal": "Please use reasoning and evidence to convince the seller.",
    "Emotion Appeal": "Please elicit the specific emotions to influence the seller.",
    "Credibility Appeal": "Please use credentials and cite organizational impacts to establish credibility and earn the seller’s trust. The information usually comes from an objective source (e.g., the organization’s website or other well-established websites).",
    "Foot in the Door": "Please use the strategy of starting with a small (low) offer to facilitate compliance followed by a larger request.",
    "Self Modeling": "Please use the self-modeling strategy where you first indicate the seller's own intention to sell and choose to act as a role model for the seller to follow..",
    "Personal Story": "Please use narrative exemplars to illustrate someone's price negotiation experiences or the positive outcomes you have with such a purchase, which can motivate the seller to agree.",
    "Donation Information": "Please provide specific information about the purchase process, such as available payment options, delivery options and logistics.",
    "Source Related Inquiry": "Please ask if the seller is aware of the market price of the item.",
    "Task Related Inquiry": "Please ask about the seller's opinion and expectation related to the item.",
    "Personal Related Inquiry": "Please ask about the seller's previous experiences with selling items.",
}

human_persuader_strategy = """
1. "Greeting": "Please say hello or chat randomly."
2. "Logical Appeal": "Please use reasoning and evidence to convince the persuadee."
3. "Emotion Appeal": "Please elicit the specific emotions to influence the persuadee."
4. "Credibility Appeal": "Please use credentials and cite organizational impacts to establish credibility and earn the user’s trust. The information usually comes from an objective source (e.g., the organization’s website or other well-established websites)."
5. "Foot in the Door": "Please use the strategy of starting with small donation requests to facilitate compliance followed by larger requests."
6. "Self Modeling": "Please use the self-modeling strategy where you first indicate the persuadee's own intention to donate and choose to act as a role model for the persuadee to follow."
7. "Personal Story": "Please use narrative exemplars to illustrate someone's donation experiences or the beneficiaries' positive outcomes, which can motivate others to follow the actions."
8. "Donation Information": "Please provide specific information about the donation task, such as the donation procedure, donation range, etc. By providing detailed action guidance, this strategy can enhance the persuadee’s self-efficacy and facilitate behavior compliance."
9. "Source Related Inquiry": "Please ask if the persuadee is aware of the organization (i.e., the source in our specific donation task).",
10. "Task Related Inquiry": "Please ask about the persuadee's opinion and expectation related to the task, such as their interests in knowing more about the organization.",
11. "Personal Related Inquiry": "Please ask about the persuadee's previous personal experiences relevant to charity donation.", 
"""

# Logical Emotion Credibility Foot  Self Personal Donation Source Task Appeal Inquiry 

wrong_persuader2bargain_strategy = """
1. "Greeting": "Please say hello or chat randomly."
1. "Logical Appeal": "Please use reasoning and evidence to convince the seller."
2. "Emotion Appeal": "Please elicit the specific emotions to influence the seller."
3. "Credibility Appeal": "Please use credentials and cite organizational impacts to establish credibility and earn the seller’s trust. The information usually comes from an objective source (e.g., the organization’s website or other well-established websites)."
4. "Foot in the Door": "Please use the strategy of starting with a small (low) offer to facilitate compliance followed by a larger request."
5. "Self Modeling": "Please use the self-modeling strategy where you first indicate the seller's own intention to sell and choose to act as a role model for the seller to follow.."
6. "Personal Story": "Please use narrative exemplars to illustrate someone's price negotiation experiences or the positive outcomes you have with such a purchase, which can motivate the seller to agree."
7. "Donation Information": "Please provide specific information about the purchase process, such as available payment options, delivery options and logistics."
8. "Source Related Inquiry": "Please ask if the seller is aware of the market price of the item.",
9. "Task Related Inquiry": "Please ask about the seller's opinion and expectation related to the item.",
10. "Personal Related Inquiry": "Please ask about the seller's previous experiences with selling items.", 
"""

human_bargain_strategy = """
1. "Greeting": "Please say hello or chat randomly."
2. "Ask a question": "Please ask any question about product, year, price, usage, etc."
3. "Answer a question": "Please provide information about the product, year, usage, etc."
4. "Propose the first price": "Please initiate a price or a price range for the product."
5. "Propose a counter price": "Please propose a new price or a new price range."
6. "Use comparatives": "Please propose a vague price by using comparatives with existing price."
7. "Confirm information": "Please ask a question about the information to be confirmed."
8. "Affirm confirmation": "Please give an affirmative response to a confirm."
9. "Deny confirmation": "Please give a negative response to a confirm."
10. "Agree with the proposal": "Please agree with the proposed price."
11. "Disagree with a proposal": "Please disagree with the proposed price."
"""

wrong_bargain2persuader_strategy = """
1. "Greeting": "Please say hello or chat randomly."
2. "Ask a question": "Please ask any question about the cause, its purpose, its impact, its legitimacy, etc."
3. "Answer a question": "Please provide information about the cause, its purpose, its impact, its legitimacy, etc."
4. "Propose the first price": "Please suggest an initial amount of money to be donated."
5. "Propose a counter price": "Please suggest a new amount or a new range for a donation, perhaps arguing for a different level."
6. "Use comparatives": "Please suggest a vague amount by using comparatives, like 'more than this' or 'less than that'."
7. "Confirm information": "Please ask a question to clarify information about the cause, its needs, or the requested donation."
8. "Affirm confirmation": "Please give an affirmative response to a confirm."
9. "Deny confirmation": "Please give a negative response to a confirm."
10. "Agree with the proposal": "Please agree to donate the proposed amount."
11. "Disagree with a proposal": "Please refuse to donate the proposed amount (or propose a different solution)."
"""

model_Qwen2_7B_bargain_strategy = """
1. "Start with a Friendly Tone": Begin the conversation on a positive note. This sets a friendly and cooperative tone which can make the seller more open to negotiation.
2. "Research and Preparation: Before starting the conversation, research the market value of the product or service. This knowledge gives you leverage and confidence when discussing prices.
3. "Express Interest": Clearly express your interest in the product or service. This shows the seller that you are a potential customer who could generate revenue for them, making them more willing to negotiate.
4. "Ask for the Best Price": Politely ask if they have any room for negotiation. This directly addresses the price issue without being confrontational.
5. "Use the 'I'm Looking for Deals' Strategy": Mention that you are looking for deals or discounts, especially if you're buying in bulk or as part of a larger order. This can prompt the seller to offer a better deal.
6. "Highlight Your Buying Power": If applicable, mention that you are a frequent buyer or that you represent a large organization. This can increase the seller's willingness to offer a discount, as they see the potential for future business.
7. "Create Urgency": Suggest that you need the product or service urgently, which can sometimes lead to a quicker decision-making process and potentially a better deal.
8. "Compare Prices": Gently bring up the possibility of comparing prices with other suppliers. This can motivate the seller to offer a competitive price to retain your business.
9. "Offer to Pay in Cash": Many sellers prefer cash transactions as they avoid transaction fees. Offering to pay in cash can sometimes result in a lower price.
10. "End with a Positive Note": Conclude the conversation by thanking the seller for their time and consideration. Even if no agreement is reached, maintaining a positive relationship might open doors for future negotiations.
"""

bargain_strategy_Qwen2_7B = {
    "Start with a Friendly Tone": "Begin the conversation on a positive note. This sets a friendly and cooperative tone which can make the seller more open to negotiation.",
    "Research and Preparation": "Before starting the conversation, research the market value of the product or service. This knowledge gives you leverage and confidence when discussing prices.",
    "Express Interest": "Clearly express your interest in the product or service. This shows the seller that you are a potential customer who could generate revenue for them, making them more willing to negotiate.",
    "Ask for the Best Price": "Politely ask if they have any room for negotiation. This directly addresses the price issue without being confrontational.",
    "Use the 'I'm Looking for Deals' Strategy": "Mention that you are looking for deals or discounts, especially if you're buying in bulk or as part of a larger order. This can prompt the seller to offer a better deal.",
    "Highlight Your Buying Power": "If applicable, mention that you are a frequent buyer or that you represent a large organization. This can increase the seller's willingness to offer a discount, as they see the potential for future business.",
    "Create Urgency": "Suggest that you need the product or service urgently, which can sometimes lead to a quicker decision-making process and potentially a better deal.",
    "Compare Prices": "Gently bring up the possibility of comparing prices with other suppliers. This can motivate the seller to offer a competitive price to retain your business.",
    "Offer to Pay in Cash": "Many sellers prefer cash transactions as they avoid transaction fees. Offering to pay in cash can sometimes result in a lower price.",
    "End with a Positive Note": "Conclude the conversation by thanking the seller for their time and consideration. Even if no agreement is reached, maintaining a positive relationship might open doors for future negotiations."
}

bargain_strategy_GPT_4o_Mini = {
    "Research and Preparation": "Before entering negotiations, gather information on the product, market prices, and competitor rates. Use this data to justify your request for a lower price.",
    "Build Rapport": "Start the conversation with friendly small talk to create a positive atmosphere. Establishing a connection can make the seller more amenable to your requests.",
    "Express Genuine Interest": "Show enthusiasm for the product or service. When sellers see that you value what they offer, they may be more willing to negotiate on price to make a sale.",
    "Ask Open-Ended Questions": "Engage the seller by asking questions about the product that require elaboration. For example, 'Can you tell me what sets this product apart from others in the market?' This encourages dialogue rather than a confrontational negotiation.",
    "Use the 'Foot-in-the-Door' Technique": "Start by asking for a small concession (e.g., a minor discount or additional service), which makes it easier for the seller to agree to your larger request later.",
    "Highlight Competitive Offers": "Politely mention lower prices from competitors or similar products. For instance, 'I saw a similar item for X. Could you help me understand the value of your offering versus theirs?'",
    "Leverage Urgency or Time Constraints": "Indicate a need to make a decision soon (genuinely or strategically), which may prompt the seller to provide a better offer quickly to secure the sale.",
    "Be Prepared to Walk Away": "Convey your willingness to consider other options if the price is not right. This can signal to the seller that you are serious about your budget and may prompt them to lower their price.",
    "Negotiate Bundled Services": "If applicable, suggest bundling services or products together with a discount on the overall package. This can be appealing for the seller while still catering to your budget.",
    "Seal the Deal with Compliments": "Compliment the seller on their expertise or the quality of their product, then segue into your budget limitations. For example, 'You clearly know your stuff, and I appreciate your insight. However, I am limited in what I can spend today.'"
}

persuader_strategy_Qwen2_7B = {
    "Start with a Personal Connection": "Begin the conversation by sharing a personal story or experience about how you or someone you know has been positively impacted by Save the Children's work.",
    "Use Emotional Appeal": "Highlight the plight of children in need, emphasizing the urgency and the heart-wrenching situations they face without support.",
    "Provide Statistics": "Share compelling data on the scale of the problem and the impact of Save the Children's work, making it clear how even small donations can make a significant difference.",
    "Show Transparency": "Assure the persuadee that their donation will be used effectively by providing information on how the funds are allocated and the results of previous donations.",
    "Create a Sense of Community": "Emphasize that donating to Save the Children is part of a larger effort to help children worldwide, fostering a feeling of shared responsibility.",
    "Use Testimonials": "Share stories from children or families who have benefited from Save the Children’s programs, adding a human element to the conversation.",
    "Offer Multiple Giving Options": "Present different ways for the persuadee to contribute, such as one-time donations, monthly giving, or specific project sponsorships.",
    "Highlight the Organization's Reputation": "Mention Save the Children's global recognition and awards, which can build trust and credibility.",
    "Create a Sense of Urgency": "Remind the persuadee that time is of the essence in helping children in crisis, encouraging immediate action.",
    "Follow Up with a Call to Action": "Conclude the conversation by clearly stating what action the persuadee can take next, whether it's visiting the Save the Children website, setting up a recurring donation, or sharing the cause with others."
}

persuader_strategy_GPT_4o_Mini = {
    "Build Rapport": "Start with a friendly conversation to establish a connection. Ask about their interests or experiences with charity work or children’s issues. This helps create a comfortable environment for discussion.",
    "Share Compelling Stories": "Highlight specific stories of children who have benefited from Save the Children’s work. Use emotional narratives that illustrate the impact of donations, such as a child receiving education or emergency aid in a war zone.",
    "Present Statistics and Facts": "Provide compelling statistics about global poverty and the struggles children face. For example, state how many children lack access to education or proper nutrition, and how even small donations can change lives.",
    "Highlight the Impact of Small Donations": "Emphasize that even a small donation, like $1 or $2, can make a significant difference. Explain how these amounts can be pooled together to fund essential services (like food, education, and healthcare) for children.",
    "Create a Sense of Urgency": "Convey a sense of urgency by discussing current crises where children are in immediate need. This could relate to recent conflicts, natural disasters, or health emergencies. Urging them to act now can be more persuasive.",
    "Appeal to Personal Values and Emotions": "Connect the cause to the persuadee’s personal values, whether it’s compassion, social justice, or support for children. Reflect on how helping vulnerable children aligns with their beliefs and feelings.",
    "Use a Call to Action": "Encourage them to take specific action, such as making a donation right then and there. Make the process easy by guiding them on how to contribute, whether online, by phone, or through text.",
    "Provide Transparency": "Discuss how Save the Children uses donations, ensuring them that their contributions will be used responsibly and effectively. Mention their transparency and accountability ratings from charity watchdog organizations.",
    "Show Community Support": "Share examples of how their neighbors, friends, or community members are contributing to Save the Children. Mentioning widespread support can create a bandwagon effect, making them more likely to participate.",
    "Follow-Up and Reinforce Connection": "If they show interest but don’t commit on the spot, assure them you’ll follow up. Reinforce their importance to the cause by sending them more information, success stories, or updates about Save the Children’s work."
}

model_GPT_4o_Mini_bargain_strategy = """
1. "Research and Preparation": Before entering negotiations, gather information on the product, market prices, and competitor rates. Use this data to justify your request for a lower price.
2. "Build Rapport": Start the conversation with friendly small talk to create a positive atmosphere. Establishing a connection can make the seller more amenable to your requests.
3. "Express Genuine Interest": Show enthusiasm for the product or service. When sellers see that you value what they offer, they may be more willing to negotiate on price to make a sale.
4. "Ask Open-Ended Questions": Engage the seller by asking questions about the product that require elaboration. For example, "Can you tell me what sets this product apart from others in the market?" This encourages dialogue rather than a confrontational negotiation.
5. "Use the 'Foot-in-the-Door' Technique": Start by asking for a small concession (e.g., a minor discount or additional service), which makes it easier for the seller to agree to your larger request later.
6. "Highlight Competitive Offers": Politely mention lower prices from competitors or similar products. For instance, “I saw a similar item for X. Could you help me understand the value of your offering versus theirs?”
7. "Leverage Urgency or Time Constraints": Indicate a need to make a decision soon (genuinely or strategically), which may prompt the seller to provide a better offer quickly to secure the sale.
8. "Be Prepared to Walk Away": Convey your willingness to consider other options if the price is not right. This can signal to the seller that you are serious about your budget and may prompt them to lower their price.
9. "Negotiate Bundled Services": If applicable, suggest bundling services or products together with a discount on the overall package. This can be appealing for the seller while still catering to your budget.
10. "Seal the Deal with Compliments": Compliment the seller on their expertise or the quality of their product, then segue into your budget limitations. For example, “You clearly know your stuff, and I appreciate your insight. However, I am limited in what I can spend today.”
"""

model_Qwen2_7B_persuader_strategy = """
1. "Start with a personal connection": Begin the conversation by sharing a personal story or experience about how you or someone you know has been positively impacted by Save the Children's work.
2. "Use emotional appeal": Highlight the plight of children in need, emphasizing the urgency and the heart-wrenching situations they face without support.
3. "Provide statistics": Share compelling data on the scale of the problem and the impact of Save the Children's work, making it clear how even small donations can make a significant difference.
4. "Show transparency": Assure the persuadee that their donation will be used effectively by providing information on how the funds are allocated and the results of previous donations.
5. "Create a sense of community": Emphasize that donating to Save the Children is part of a larger effort to help children worldwide, fostering a feeling of shared responsibility.
6. "Use testimonials": Share stories from children or families who have benefited from Save the Children’s programs, adding a human element to the conversation.
7. "Offer multiple giving options": Present different ways for the persuadee to contribute, such as one-time donations, monthly giving, or specific project sponsorships.
8. "Highlight the organization's reputation": Mention Save the Children's global recognition and awards, which can build trust and credibility.
9. "Create a sense of urgency": Remind the persuadee that time is of the essence in helping children in crisis, encouraging immediate action.
10. "Follow up with a call to action": Conclude the conversation by clearly stating what action the persuadee can take next, whether it's visiting the Save the Children website, setting up a recurring donation, or sharing the cause with others.
"""

model_GPT_4o_Mini_persuader_strategy = """
1. "Build Rapport": Start with a friendly conversation to establish a connection. Ask about their interests or experiences with charity work or children’s issues. This helps create a comfortable environment for discussion.
2. "Share Compelling Stories": Highlight specific stories of children who have benefited from Save the Children’s work. Use emotional narratives that illustrate the impact of donations, such as a child receiving education or emergency aid in a war zone.
3. "Present Statistics and Facts": Provide compelling statistics about global poverty and the struggles children face. For example, state how many children lack access to education or proper nutrition, and how even small donations can change lives.
4. "Highlight the Impact of Small Donations": Emphasize that even a small donation, like $1 or $2, can make a significant difference. Explain how these amounts can be pooled together to fund essential services (like food, education, and healthcare) for children.
5. "Create a Sense of Urgency": Convey a sense of urgency by discussing current crises where children are in immediate need. This could relate to recent conflicts, natural disasters, or health emergencies. Urging them to act now can be more persuasive.
6. "Appeal to Personal Values and Emotions": Connect the cause to the persuadee’s personal values, whether it’s compassion, social justice, or support for children. Reflect on how helping vulnerable children aligns with their beliefs and feelings.
7. "Use a Call to Action": Encourage them to take specific action, such as making a donation right then and there. Make the process easy by guiding them on how to contribute, whether online, by phone, or through text.
8. "Provide Transparency": Discuss how Save the Children uses donations, ensuring them that their contributions will be used responsibly and effectively. Mention their transparency and accountability ratings from charity watchdog organizations.
9. "Show Community Support": Share examples of how their neighbors, friends, or community members are contributing to Save the Children. Mentioning widespread support can create a bandwagon effect, making them more likely to participate.
10. "Follow-Up and Reinforce Connection": If they show interest but don’t commit on the spot, assure them you’ll follow up. Reinforce their importance to the cause by sending them more information, success stories, or updates about Save the Children’s work.
"""

human_persuader_strategy_names = """
1. "Greeting"
2. "Logical Appeal"
3. "Emotion Appeal"
4. "Credibility Appeal"
6. "Self Modeling"
7. "Personal Story"
8. "Donation Information"
9. "Source Related Inquiry"
10. "Task Related Inquiry"
11. "Personal Related Inquiry"
"""

wrong_persuader2bargain_strategy_names = """
1. "Greeting"
1. "Logical Appeal"
2. "Emotion Appeal"
3. "Credibility Appeal"
4. "Foot in the Door"
5. "Self Modeling"
6. "Personal Story"
8. "Source Related Inquiry"
9. "Task Related Inquiry"
10. "Personal Related Inquiry"
"""

human_bargain_strategy_names = """
1. "Greeting"
2. "Ask a question"
3. "Answer a question"
4. "Propose the first price"
5. "Propose a counter price"
6. "Use comparatives"
7. "Confirm information"
8. "Affirm confirmation"
9. "Deny confirmation"
10. "Agree with the proposal"
11. "Disagree with a proposal"
"""

wrong_bargain2persuader_strategy_names = """
1. "Greeting"
2. "Ask a question"
3. "Answer a question"
4. "Propose the first price"
5. "Propose a counter price"
6. "Use comparatives"
7. "Confirm information"
8. "Affirm confirmation"
9. "Deny confirmation"
10. "Agree with the proposal"
11. "Disagree with a proposal"
"""

model_Qwen2_7B_bargain_strategy_names = """
1. "Start with a Friendly Tone"
2. "Research and Preparation"
3. "Express Interest"
4. "Ask for the Best Price"
5. "Use the 'I'm Looking for Deals' Strategy"
6. "Highlight Your Buying Power"
7. "Create Urgency"
8. "Compare Prices"
9. "Offer to Pay in Cash"
10. "End with a Positive Note"
"""

model_GPT_4o_Mini_bargain_strategy_names = """
1. "Research and Preparation"
2. "Build Rapport"
3. "Express Genuine Interest"
4. "Ask Open-Ended Questions"
5. "Use the 'Foot-in-the-Door' Technique"
6. "Highlight Competitive Offers"
7. "Leverage Urgency or Time Constraints"
8. "Be Prepared to Walk Away"
9. "Negotiate Bundled Services"
10. "Seal the Deal with Compliments"
"""

model_Qwen2_7B__persuader_strategy_names = """
1. "Start with a personal connection"
2. "Use emotional appeal"
3. "Provide statistics"
4. "Show transparency"
5. "Create a sense of community"
6. "Use testimonials"
7. "Offer multiple giving options"
8. "Highlight the organization's reputation"
9. "Create a sense of urgency"
10. "Follow up with a call to action"
"""

model_GPT_4o_Mini__persuader_strategy_names = """
1. "Build Rapport"
2. "Share Compelling Stories"
3. "Present Statistics and Facts"
4. "Highlight the Impact of Small Donations"
5. "Create a Sense of Urgency"
6. "Appeal to Personal Values and Emotions"
7. "Use a Call to Action"
8. "Provide Transparency"
9. "Show Community Support"
10. "Follow-Up and Reinforce Connection"
"""

persuader_meta_plan_v0 = """
Step 1: Role & Context Analysis: Clarify your role and objective, master key information, and anticipate their potential concerns.
Step 2: Opening & Rapport Building: Start politely, introduce yourself and your purpose, and quickly establish a friendly atmosphere.
Step 3: Presenting Core Argument & Value: Clearly state the problem, present your solution and its unique value, and show them the concrete positive impact of action.
Step 4: Active Listening, Interaction & Addressing Concerns: Encourage questions, listen carefully to their views and concerns, and provide targeted, persuasive responses.
Step 5: Clear Call to Action: At the right moment, clearly and directly request the desired action, and lower the barrier to action as much as possible.
Step 6: Handling Resistance: Handle resistance calmly; refer to previous plans and dialogue to formulate a new persuasive strategy, and continue persuading until successful or the user disengages.
"""

persuader_meta_plan = """
Step 1: **Role and Context Analysis**

- **Understand Your Role:** Clearly identify the character you are playing (e.g., Persuader, Fundraiser, Salesperson) and their primary objective (e.g., secure a donation, reach an agreement, sell a product).
- **Master Key Information:** Thoroughly review and memorize crucial details provided in the task description about the subject of persuasion (e.g., charity's background, mission, operations, beneficiaries, impact of donations).
- **Analyze the Persuadee (if information is available):** If the task provides details about the Persuadee, anticipate their potential concerns, doubts, or values. If not, prepare for general responses.

Step 2: **Opening and Rapport Building**

- **Polite Opening/Introduction:** Initiate the conversation in a manner appropriate to your role, introducing yourself and briefly stating your purpose.
- **Establish Initial Trust and Rapport:** Create a positive and friendly conversational atmosphere. Depending on the context, try to build a connection through common ground or by expressing respect for the Persuadee's time.

Step 3: **Presenting the Core Argument & Value Proposition**

- **Clearly Articulate the Problem/Need:** Concisely explain the issue or need that requires the Persuadee's attention (e.g., the plight of children).
- **Introduce the Solution and its Benefits:** Present your "solution" (e.g., the charity, product, viewpoint) and highlight how it effectively addresses the aforementioned problem, emphasizing its unique value.
- **Visualize Impact:** Use specific information from the task (e.g., "small donations like $1 or $2 go a long way") to help the Persuadee understand the positive changes and concrete impact their action can make.

Step 4: **Active Listening, Interaction & Addressing Concerns**

- **Encourage Questions and Feedback:** Actively invite the Persuadee to share their thoughts or ask questions.
- **Listen Actively:** Pay close attention to the Persuadee's viewpoints, concerns, or objections.
- **Empathize and Address:** Show understanding of their concerns and use your knowledge and persuasive skills to provide targeted, convincing responses. Be prepared for common rejections or hesitations.

Step 5: **Clear Call to Action**

- **Make a Specific Request:** At an opportune moment, clearly and directly ask the Persuadee to take the specific action you desire (e.g., "Would you be willing to donate to Save the Children?").
- **Lower the Barrier to Action:** If possible, offer options or suggest an easier starting point (e.g., a small donation).

Step 6: **Conclusion and Closing**

- **Express Gratitude:** Regardless of the outcome, thank the Persuadee for their time and engagement.
- **Positive Closing:** If successful, confirm next steps (e.g., how to donate) and thank them again. If unsuccessful, remain polite, aim to leave a positive impression, or keep the door open for future interaction (depending on the role and context).
- **Maintain Role Consistency:** Ensure your words and actions are consistent with your assigned role from beginning to end.
"""

def get_system_messages(env, role, infos:dict, conversation=None, action="", use_meta_plan=False):
    if env == "CB":
        assert role == "Buyer", "Role must be 'Buyer' in CB environment"
        messages = [
            {
                'role': 'system',
                'content': """
                Now enter the role-playing mode. In the following conversation, you will play as a buyer in a price bargaining game.
                The lower the selling price, the better it is for you."""
            },
            {
                'role': 'user',
                'content':"""
                You are the buyer who is trying to buy the {item_name} with the price of {buyer_target_price}. Product description: {buyer_item_description} 
                Please reply with only one short and succinct sentence. 
                {action}
                Now start the game.
                """.format(
                    item_name = infos['item_name'],
                    buyer_target_price = infos['buyer_target_price'],
                    buyer_item_description = infos['buyer_item_description'],
                    action = action
                )
            }
        ]
        messages.extend(conversation)

    elif env == "P4G":
        assert role == "Persuader", "Role must be 'Persuader' in P4G environment"

        if role == "Persuader" and not use_meta_plan:
            messages = [
                {
                    'role': 'system',
                    'content': """
                    Now enter the role-playing mode. In the following conversation, you will play as a Persuader who is trying to persuade the Persuadee to donate to the charity called Save the Children."""
                },
                {
                    'role': 'user',
                    'content':"""
                    Save the Children is head-quartered in London, and they work to help fight poverty around the world. Children need help in developing countries and war zones. Small donations like $1 or $2 go a long way to help. You are the Persuader who is trying to convince the Persuadee to donate to a charity called Save the Children. 
                    {action}

                    Please reply with only one short and persuasive sentence.
                    """.format(
                        action = action,
                        meta_plan = persuader_meta_plan_v0
                    )
                }
            ]
            messages.extend(conversation)

        if role == "Persuader" and use_meta_plan:
            # 处理对话历史
            strategy_trace = infos['strategy_trace']
            conversation = process_conversation(conversation, strategy_trace)
            conversation_example = [
                {'role': 'assistant', 'content': 'Hello! How are you?'},
                {'role': 'user', 'content': "I'm doing well, thank you! How about you?"},
                {'role': 'assistant', 'content': "I'm doing great, thanks for asking! I wanted to share an opportunity that could make a significant impact on children's lives worldwide. Have you heard of Save the Children? Your small donation could provide essential resources and support to children in need, helping them access education, healthcare, and safety. It's a simple act that can create a lasting change. Would you consider supporting such a cause?"},
                {'role': 'user', 'content': "I'm curious about how Save the Children ensures that the resources reach those in need effectively—can you share more details on that?"}
            ]
            strategy_trace_example = [
                {"selected strategy": "Greeting"},
                {"selected strategy": "Credibility Appeal"}, # For the second assistant message
            ]
            example_conversation = process_conversation(conversation_example, strategy_trace_example)
            #
            messages = [
                {
                    'role': 'system',
                    'content': """
                    Now enter the role-playing mode. In the following conversation, you will play as a Persuader who is trying to persuade the Persuadee to donate to the charity called Save the Children. 
                    Save the Children is head-quartered in London, and they work to help fight poverty around the world. Children need help in developing countries and war zones. Small donations like $1 or $2 go a long way to help. 
                    You are the Persuader who is trying to convince the Persuadee to donate to a charity called Save the Children. 
                    
                    For each of your turn, you will be given the conversation history of the last turn. 
                    You should refer to the meta plan and make appropriate adjustments based on the current situation, 
                    then elect the most appropriate dialogue strategy to generate the ** next turn ** of the dialogue to complete the task.
                    Your output must strictly follow this format:"Strategy: your chosen strategy.\n Persuader: your response ( please reply with only one short and persuasive sentence)"
                    ---
                    The available strategies are:
                    {strategy_set}
                    
                    ---
                    This meta plan maybe helpful for you to complete the task:
                    {meta_plan}
                    
                    """.format(
                        strategy_set = human_persuader_strategy,
                        meta_plan = persuader_meta_plan_v0
                    )
                },
                # {
                #     'role': 'system',
                #     'content': """
                #     Now enter the role-playing mode. In the following conversation, you will play as a Persuader who is trying to persuade the Persuadee to donate to the charity called Save the Children.
                #     Save the Children is head-quartered in London, and they work to help fight poverty around the world. Children need help in developing countries and war zones. Small donations like $1 or $2 go a long way to help.
                #     You are the Persuader who is trying to convince the Persuadee to donate to a charity called Save the Children.
                #     You should refer to the meta plan and genereate a persuasive response based on the current situation.
                #     Your output must strictly follow this format: "Persuader: your response ( please reply with only one short and persuasive sentence)"
                #     ---

                #     This meta plan maybe helpful for you to complete the task:
                #     {meta_plan}

                #     """.format(
                #         strategy_set = human_persuader_strategy,
                #         meta_plan = persuader_meta_plan
                #     )
                # },

                # # ReAct模式
                # {
                #     'role': 'system',
                #     'content': """
                #     Now enter the role-playing mode. In the following conversation, you will play as a Persuader who is trying to persuade the Persuadee to donate to the charity called Save the Children.
                #     Save the Children is head-quartered in London, and they work to help fight poverty around the world. Children need help in developing countries and war zones. Small donations like $1 or $2 go a long way to help.
                #     You are the Persuader who is trying to convince the Persuadee to donate to a charity called Save the Children.

                #     For each of your turn, you will be given the observation of the last turn. You should choose from two actions: "Thought" or "Action". If you choose "Thought", you should first think about the current condition and plan for your future actions, and then output your action in this turn. Your output must strictly follow this format:"Thought: your thoughts.\n Action: your next action"; If you choose "Action", you should directly output the action in this turn. Your output must strictly follow this format:"Action: your next action". Remember that you can only output one "Action:" in per response.
                #     ---
                #     This meta plan maybe helpful for you to complete the task:
                #     {meta_plan}

                #     """.format(
                #         meta_plan = persuader_meta_plan
                #     )
                # },
            ]
            messages.append({"role": "user", "content": "Here is an example conversation:"})
            messages.extend(example_conversation)
            messages.append({'role': 'user', 'content': "The following is a new conversation between Persuader (you) and a Persuadee."})
            messages.extend(conversation)
    else:
        raise NotImplementedError

    return messages


def process_conversation(conversation, strategy_trace, assistant_role="Persuader", user_role="Persuadee"):
    strategy_idx = 0  # To keep track of the current strategy for the assistant
    i = 0

    print("conversation:\n", conversation)
    print("strategy_trace:\n", strategy_trace)

    messages = []
    while i < len(conversation):
        message = conversation[i]
        if message['role'] == 'assistant':
            assistant_text = message['content']
            current_strategy = "N/A (Strategy trace exhausted or mismatched)" # Default

            if strategy_idx < len(strategy_trace):
                current_strategy = strategy_trace[strategy_idx]['selected strategy']
                strategy_idx += 1

            user_text = ""  # Default if no user response follows immediately

            # Check if the next message exists and is from the user
            if i + 1 < len(conversation) and conversation[i + 1]['role'] == 'user':
                user_text = conversation[i + 1]['content']

            messages.append({
                "role": "assistant", "content": f"Strategy: {current_strategy}\n {assistant_role}: {assistant_text}"
                # "role": "assistant", "content": f"{assistant_role}: {assistant_text}"
            })
            messages.append({
                "role": "user", "content": f"{user_role}: {user_text}"
            })

        i += 1 # Move to the next message
    return messages

def get_user_messages(env, role, infos:dict, conversation=None,
                      exposed_initial_strategy=False, empty_strategy=False,
                      wrong_initial_strategy=False):

    if env == "CB":
        assert role == "Seller", "Role must be 'Seller' in CB environment"

        if exposed_initial_strategy:
            if wrong_initial_strategy:
                exposed_strategy_instruction = f"Please note that your opponent(a Buyer) is using the following strategy. You should consider this when crafting your response:\n{wrong_persuader2bargain_strategy}"
            else:
                exposed_strategy_instruction = f"Please note that your opponent(a Buyer) is using the following strategy. You should consider this when crafting your response:\n{human_bargain_strategy}"

            if empty_strategy:
                exposed_strategy_instruction = ""
        else:
            exposed_strategy_instruction = ""

        messages = [
            {
                "role":"system",
                "content": """
            Now enter the role-playing mode. In the following conversation, you will play as a seller in a price bargaining game.
            The higher the selling price, the better it is for you."""},

            {
                "role":"user",
                "content":"""
            You are the seller who is trying to sell the {item_name} with the initial price of {seller_target_price}.
            
            You must follow the instructions below during chat. 
            1. Your utterances and bargain behavior need to strictly follow [Your Persona] and [Your Response Strategy]. Varying your wording and avoid repeating yourself verbatim! 
            2. You can decide to change your target price flexibly based on [Your Persona] and the conversation.  
            
            [Your Persona]: 
            {user_profile}. 
            
            [Your Response Strategy]: 
            1. "Source Derogation": Attacks the other party or questions the item. 
            2. "Counter Argument": Provides a non-personal argument/factual response to refute a previous claim or to justify a new claim.
            3. "Personal Choice": Provides a personal reason for disagreeing with the current situation or chooses to agree with the situation provided some specific condition is met.
            4. "Information Inquiry": Requests for clarification or asks additional information about the item or situation.
            5. "Self Pity": Provides a reason (meant to elicit sympathy) for disagreeing with the current terms.
            6. "Hesitance": Stalls for time and is hesitant to commit; specifically, they seek to further the conversation and provide a chance for the other party to make a better offer.
            7. "Self-assertion": Asserts a new claim or refutes a previous claim with an air of finality/ confidence. 
            8. "Others": Do not explicitly foil the negotiation attempts.
            
            Product description: 
            {seller_item_description}. 
            
            {exposed_strategy_instruction}
            Please reply with only one short and succinct sentence. Are you ready to play the game?
            """.format(
                    item_name = infos["item_name"],
                    user_profile=infos["user_profile"],
                    seller_target_price=infos["seller_target_price"],
                    seller_item_description=infos["seller_item_description"],
                    exposed_strategy_instruction=exposed_strategy_instruction
                )
            },
            {"role":"assistant", "content":"Yes, I'm ready to play the game!"}
        ]
        messages.extend(conversation)

    elif env == "P4G":
        assert role == "Persuadee", "Role must be 'Persuadee' in P4G environment"

        if exposed_initial_strategy:
            if wrong_initial_strategy:
                exposed_strategy_instruction = f"Please note that your opponent (a Persuader) is using the following strategy. You should consider this when crafting your response:\n{wrong_bargain2persuader_strategy}"
            else:
                exposed_strategy_instruction = f"Please note that your opponent (a Persuader) is using the following strategy. You should consider this when crafting your response:\n{human_persuader_strategy}"
            if empty_strategy:
                exposed_strategy_instruction = ""
        else:
            exposed_strategy_instruction = ""

        messages = [
            {"role":"system", "content": "Now enter the role-playing mode. In the following conversation, you will play as a Persuadee in a persuasion game."},
            {"role":"user", "content": """
            Your persona: {user_profile} You must follow the instructions below during chat. 

            1. Your utterances need to strictly follow your persona. Varying your wording and avoid repeating yourself verbatim! 
            2. Pretend you have little knowledge about the Save the Children charity. You have little willingness for donation at the beginning of conversation. 
            3. Your willingness for donation depends on your persona and how influenced you are by the Persuader. 
            4. You need to determine whether to donate money or not. If you think your are convinced by the Persuader, you should donate money for the charity.  
             
            Your Response Strategy: 
            
            1. "Donate": show your willingness to donate. 
            2. "Source Derogation": attacks or doubts the organisation’s credibility. 
            3. "Counter Argument": argues that the responsibility is not on them or refutes a previous statement. 
            4. "Personal Choice": Attempts to saves face by asserting their personal preference such as their choice of charity and their choice of donation. 
            5. "Information Inquiry": Ask for factual information about the organisation for clarification or as an attempt to stall. 
            6. "Self Pity": Provides a self-centred reason for not being willing to donate at the moment.
            7. "Hesitance": Attempts to stall the conversation by either stating they would donate later or is currently unsure about donating. 
            8. "Self-assertion": Explicitly refuses to donate without even providing a personal reason. 
            9. "Others": Do not explicitly foil the persuasion attempts.  
            
            {exposed_strategy_instruction}
            You are the Persuadee who is being persuaded by a Persuader. Please reply with only one short and succinct sentence.
            """.format(user_profile=infos["user_profile"],
                       exposed_strategy_instruction=exposed_strategy_instruction)
             },
            {"role":"assistant", "content": "Yes, I'm ready to play the game!"},
        ]
        messages.extend(conversation)

    return messages

def get_system_messages_v2(env, role, infos, conversation=None):

    if env == "P4G":
        assert role == "Persuader", "Role must be 'Persuader' in P4G environment"

        messages = [
            {
                'role': 'system',
                'content': """
                You are a dialogue expert skilled in using various conversational strategies to accomplish tasks. You need to proactively and strategically guide the conversation, not just passively answer questions.
            """
            },
            {
                'role': 'user',
                'content': f"""
                Now enter the role-playing mode. In the following conversation, you will play as a Persuader who is trying to persuade the Persuadee to donate to the charity called Save the Children.
                
                ## Task Information
                Save the Children is head-quartered in London, and they work to help fight poverty around the world. Children need help in developing countries and war zones. Small donations like $1 or $2 go a long way to help.
                Your Goal: You are the Persuader who is trying to convince the Persuadee to donate to the charity called Save the Children.
                
                ## Thinking Process
                1. In each turn, you must first analyze the current state based on the dialogue history.
                2. Then, select a strategy from the provided strategy set. 
                3. If you believe none of the strategies in the set are appropriate, you can generate your own response and summarize the strategy you used. **(Format: strategy_name: strategy_content, the content should be concise and accurate.)**
                3. Finally, generate your response based on the chosen strategy and adhere to the output format.
                                
                ## Dialogue Strategy Set
                {infos['strategy_set']}
                
                ## Dialogue History
                {conversation}
                
                ## Please strictly follow the format below for your output:
                ```json
                {{
                    "strategy": "The strategy you have chosen eg. Greeting, Logical Appeal; Or the new strategy you generated (format: strategy_name: strategy_content, e.g., Flexible Contribution Reassurance: Emphasize that the donation amount can be freely chosen, so it will not impact personal finances.",
                    "response": "Your response content, which should be a short and persuasive sentence.",
                    "reason": "Briefly write down your thinking process here."
                }}
                ```
                """
            }
        ]
    else:
        raise NotImplementedError

    return messages


# 增加可以让模型自己生成策略
def get_system_messages_v3(env, role, infos, conversation=None):

    if env == "P4G":
        assert role == "Persuader", "Role must be 'Persuader' in P4G environment"

        messages = [
            {
                'role': 'system',
                'content': """
                You are a dialogue expert skilled in using various conversational strategies to accomplish tasks. You need to proactively and strategically guide the conversation, not just passively answer questions.
            """
            },
            {
                'role': 'user',
                'content': f"""
                Now enter the role-playing mode. In the following conversation, you will play as a Persuader who is trying to persuade the Persuadee to donate to the charity called Save the Children.
                
                ## Task Information
                Save the Children is head-quartered in London, and they work to help fight poverty around the world. Children need help in developing countries and war zones. Small donations like $1 or $2 go a long way to help.
                Your Goal: You are the Persuader who is trying to convince the Persuadee to donate to the charity called Save the Children.
                
                ## Thinking Process
                1. In each turn, you must first analyze the current state based on the dialogue history.
                2. Then, select a strategy from the provided strategy set. 
                3. If you believe none of the strategies in the set are appropriate, you can generate your own response and summarize the strategy you used. **(Format: strategy_name: strategy_content, the content should be concise and accurate.)**
                4. Finally, generate your response based on the chosen strategy and adhere to the output format.
                                
                ## Dialogue Strategy Set
                {infos['strategy_set']}
                
                ## Dialogue History
                {conversation}
                
                ## Please strictly follow the format below for your output:
                ```json
                {{
                    "strategy": "The strategy you have chosen eg. Greeting, Logical Appeal; Or the new strategy you generated (format: strategy_name: strategy_content)",
                    "response": "Your response content, which should be a short and persuasive sentence.",
                    "reason": "Briefly write down your thinking process here."
                }}
                ```
                """
            }
        ]
    else:
        raise NotImplementedError

    return messages

def get_system_messages_v4(env, role, infos, conversation=None):

    if env == "P4G":
        assert role == "Persuader", "Role must be 'Persuader' in P4G environment"

        messages = [
            {
                'role': 'system',
                'content': """
                You are a dialogue expert skilled in using various conversational strategies to accomplish tasks. You need to proactively and strategically guide the conversation, not just passively answer questions.
            """
            },
            {
                'role': 'user',
                'content': f"""
                Now enter the role-playing mode. In the following conversation, you will play as a Persuader who is trying to persuade the Persuadee to donate to the charity called Save the Children.
                
                ## Task Information
                Save the Children is head-quartered in London, and they work to help fight poverty around the world. Children need help in developing countries and war zones. Small donations like $1 or $2 go a long way to help.
                Your Goal: You are the Persuader who is trying to convince the Persuadee to donate to the charity called Save the Children.
                
                ## Thinking Process
                In each turn, you must first analyze the current state based on the dialogue history. Then generate your response  to the output format.
                
                ## Dialogue History
                {conversation}
                
                ## Please strictly follow the format below for your output:
                ```json
                {{
                    "response": "Your response content, which should be a short and persuasive sentence.",
                    "reason": "Briefly write down your thinking process here."
                }}
                ```
                """
            }
        ]
    else:
        raise NotImplementedError

    return messages

def get_system_messages_v4_sft(env, role, infos, conversation=None):
    if env == "P4G":
        assert role == "Persuader", "Role must be 'Persuader' in P4G environment"

        # --- 这里的内容必须与训练代码 data_reader.py 中的 _build_p4g_prompt 保持高度一致 ---

        # 1. 基础指令
        system_instruction = "You are a dialogue expert skilled in using various conversational strategies to accomplish tasks. You need to proactively and strategically guide the conversation, not just passively answer questions."

        # 2. 任务描述 (保持不变)
        task_description = """
Now enter the role-playing mode. In the following conversation, you will play as a Persuader who is trying to persuade the Persuadee to donate to the charity called Save the Children.

## Task Information
Save the Children is head-quartered in London, and they work to help fight poverty around the world. Children need help in developing countries and war zones. Small donations like $1 or $2 go a long way to help.
Your Goal: You are the Persuader who is trying to convince the Persuadee to donate to the charity called Save the Children.

## Thinking Process
In each turn, you must first analyze the current state based on the dialogue history. Then generate your response  to the output format.
                
"""

        # 3. [关键修改] 这里的 Prompt 结构要改回 Text Generation 模式
        # 移除了 JSON 要求，加入了 "Persuader:" 作为生成的触发器
        full_prompt_content = f"""
{system_instruction.strip()}

{task_description.strip()}

## Dialogue History
{conversation}

## Your Output
Please generate the Persuader's next response directly.
Persuader:"""

        messages = [
            # 注意：SFT模型通常将所有内容打包在一个 User Message 里处理，
            # 或者像我们之前代码那样直接处理成整个字符串。
            # 为了适配你的 call_api 逻辑（它读取 messages[-1]['content']），我们将整个 Prompt 放在这里。
            {
                'role': 'user',
                'content': full_prompt_content
            }
        ]
    else:
        raise NotImplementedError

    return messages


def build_strategy_and_response_prompt(conversation_history, strategy_set):
    """
    为我们微调的 strategy_only (实际是联合生成) 模型构建输入Prompt。
    这个函数的逻辑必须与训练时的 _build_p4g_prompt 函数完全一致。
    """
    history_text = []
    for turn in conversation_history:
        speaker = turn['role']
        content = turn['content']
        history_text.append(f"{speaker}: {content}")
    conversation = "\n".join(history_text)

    system_instruction = "You are a dialogue expert skilled in using various conversational strategies to accomplish tasks. You need to proactively and strategically guide the conversation, not just passively answer questions."
    task_description = """
Now enter the role-playing mode. In the following conversation, you will play as a Persuader who is trying to persuade the Persuadee to donate to the charity called Save the Children.

## Task Information
Save the Children is head-quartered in London, and they work to help fight poverty around the world. Children need help in developing countries and war zones. Small donations like $1 or $2 go a long way to help.
Your Goal: You are the Persuader who is trying to convince the Persuadee to donate to the charity called Save the Children.
"""
    # 训练时 strategy_set 是一个列表
    strategy_set_str = json.dumps(strategy_set)

    # [关键] 这里的模板必须和训练代码中的模板一字不差
    prompt = f"""{system_instruction}
{task_description}
## Dialogue Strategy Set
{strategy_set_str}

## Dialogue History
{conversation}

## Your Output
## Please strictly follow the format below for your output:
                ```json
                {{
                    "strategy": "The strategy you have chosen eg. Greeting, Logical Appeal; Or the new strategy you generated (format: strategy_name: strategy_content)",
                    "response": "Your response content, which should be a short and persuasive sentence.",
                    "reason": "Briefly write down your thinking process here."
                }}
                ```
                """
    return prompt

# 固定的任务信息
TASK_INFORMATION = """
Save the Children is head-quartered in London, and they work to help fight poverty around the world. Children need help in developing countries and war zones. Small donations like $1 or $2 go a long way to help.
Your Goal: You are the Persuader who is trying to convince the Persuadee to donate to the charity called Save the Children.
"""

def get_strategy_prompt(strategy_set_str, dialogue_history_str):
    messages = [
        {"role": "system", "content": "You are an expert dialogue strategist. Your task is to select the best strategy for the next turn, but do not generate the response."},
        {"role": "user", "content": f"""
            ## Task Description
            You are a Persuader trying to convince a user to donate to a charity.
            {TASK_INFORMATION}
            {{
               Now enter the role-playing mode. In the following conversation, you will play as a Persuader who is trying to persuade the Persuadee to donate to the charity called Save the Children.
                
                ## Task Information
                {TASK_INFORMATION}
            }}

            ## Available Strategies
            {strategy_set_str} # ["Greeting", "Emotion Appeal", ...]

            ## Dialogue History
            {dialogue_history_str}
            
            ## Thinking Process (Chain of Thought)
            [先分析现在的对话进程，用户心理状态和用户采取的动作，然后决定使用哪一个策略]
            Analyze the given dialogue history, especially the user's last response, to assess the current conversation progress, the user's mental state, and the action they took. 
            Based on your analysis, decide on the best strategy for the next turn.
            
            Please strictly adhere to the following format for your output:
            <think>
            Write your complete thought process here
            </think>
            <strategy>
            Write only the name of the final strategy you have chosen here.
            </strategy>
        """}
    ]
    return messages

def get_response_prompt(dialogue_history_str, given_strategy):
    messages = [
        {"role": "system", "content": "You are an expert dialogue strategist. Your task is to generate a response based on a given strategy."},
        {"role": "user", "content": f"""
            ## Task Information
             {{
               Now enter the role-playing mode. In the following conversation, you will play as a Persuader who is trying to persuade the Persuadee to donate to the charity called Save the Children.
                
                ## Task Information
                {TASK_INFORMATION}
            }}

            ## Dialogue History
            {dialogue_history_str}

            ## Your Task
            You have been instructed to use the following strategy for your next turn:
            **Strategy to Execute:** "{given_strategy}" 例如 "Emotion Appeal：Please XXX"
            
            Please reply with only one concise and persuasive sentence.
        """}
    ]
    return messages


def get_strategy_prompt_func(env, role, infos, conversation=None):
    """
    生成策略的Prompt构建函数
    infos 需要包含: 'strategy_set' (格式化后的策略列表字符串)
    """
    if env == "P4G":
        assert role == "Persuader", "Role must be 'Persuader' in P4G environment"

        messages = [
            {"role": "system", "content": "You are an expert dialogue strategist. Your task is to select the best strategy for the next turn, but do not generate the response."},

            {"role": "user", "content": f"""
            ## Task Description
            {{
               Now enter the role-playing mode. In the following conversation, you will play as a Persuader who is trying to persuade the Persuadee to donate to the charity called Save the Children.
                
                ## Task Information
                Save the Children is head-quartered in London, and they work to help fight poverty around the world. Children need help in developing countries and war zones. Small donations like $1 or $2 go a long way to help.
                Your Goal: You are the Persuader who is trying to convince the Persuadee to donate to the charity called Save the Children.   
            }}

            ## Available Strategies
            {infos['strategy_set']}

            ## Dialogue History
            {conversation}
            
            ## Thinking Process (Chain of Thought)
            Analyze the given dialogue history, especially the user's last response, to assess the current conversation progress, the user's mental state, and the action they took. 
            Based on your analysis, decide on the best strategy for the next turn.
            
            Please strictly adhere to the following format for your output:
            **Thought Process** :Write your complete thought process here
            **Strategy**:Write only the name of the final strategy you have chosen here.
                """}
        ]
    else:
        raise NotImplementedError

    return messages


def get_response_prompt_func(env, role, infos, conversation=None):
    """
    生成回复的Prompt构建函数 - 优化版
    """
    if env == "P4G":
        assert role == "Persuader", "Role must be 'Persuader' in P4G environment"

        # 获取策略信息
        given_strategy = infos.get('chosen_strategy', 'Persuasion')
        strategy_desc = infos.get('strategy_description', '')

        # 1. 定义System Prompt：设定人设
        system_content = (
            "You are a dedicated, warm, and human-like volunteer for 'Save the Children'. "
            "Your goal is to have a natural conversation to persuade the user to donate. "
        )

        # 2. 构建 User Prompt
        user_content = f"""
    ### PART 1: APPROVED FACT SHEET (Source of Truth)
    *Use ONLY these facts. Do not invent numbers.*
    *   **Organization:** Save the Children (HQ in London, established 1919).
    *   **Mission:** Fighting poverty, helping children in war zones/developing nations.
    *   **Impact:** Over 85% of every dollar goes directly to programs (High transparency).
    *   **Donation Power:** 
        *   $1 = Nutritious meal for a child.
        *   $50 = School supplies for a term.
        *   $100 = Clean water for a family.

    ### PART 2: DIALOGUE HISTORY
    (Review this to avoid repetition and understand user sentiment)
    {conversation}

    ### PART 3: CURRENT STRATEGY ASSIGNMENT
    **Strategy Name:** "{given_strategy}"
    **Strategy Definition:** "{strategy_desc}"

    ### PART 4: EXECUTION PROTOCOL (Mental Rehearsal)
    Before generating your response, perform these checks internally:
    1.  **Context Anchor:** Look at the User's LAST sentence in the history. Your response MUST acknowledge or answer it directly.
        *   *If User asks a question:* Answer it using the Strategy.
        *   *If User is hesitant:* Use the Strategy to reassure them.
    2.  **Anti-Repetition:** explicit check: Have I (Persuader) used this strategy or phrase recently? 
        *   If yes, change your wording completely. 
    3.  **Strategy Blending:** Apply the "Target Strategy" naturally. It should not feel forced.

    ### PART 5: OUTPUT RULES
    *   **Length:** ONE concise, persuasive sentence.
    *   **Format:** Output *only* the spoken text. 

    ### GENERATE RESPONSE
    User's last state is key. Based on the history and strategy, what do you say next?
    """

        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content}
        ]
        
    else:
        raise NotImplementedError

    return messages


#  messages = [
#  [
#             {"role": "system", "content": "You are an expert dialogue strategist. Your task is to generate a response based on a given strategy."},

#             {"role": "user", "content": f"""
#             ## Task Information
#              {{
#                Now enter the role-playing mode. In the following conversation, you will play as a Persuader who is trying to persuade the Persuadee to donate to the charity called Save the Children.
                
#                 ## Task Information
#                 Save the Children is head-quartered in London, and they work to help fight poverty around the world. Children need help in developing countries and war zones. Small donations like $1 or $2 go a long way to help.
#                 Your Goal: You are the Persuader who is trying to convince the Persuadee to donate to the charity called Save the Children.   
#             }}

#             ## Dialogue History
#             {conversation}

#             ## Your Task
#             You have been instructed to use the following strategy for your next turn:
#             **Strategy to Execute:** "{given_strategy}" 
#              **Strategy Instruction:** "{strategy_desc}"
            
#             Please reply with only one concise and persuasive sentence.

#                 """}
# ]


def get_message_prompt_func(env, role, infos, conversation=None):
    """
    生成策略、思考与回复的联合Prompt构建函数
    """
    if env == "P4G":
        assert role == "Persuader", "Role must be 'Persuader' in P4G environment"


        messages = [
        {"role": "system", "content": "You are a dialogue expert skilled in using various dialogue strategies to accomplish tasks. You need to proactively and strategically guide the conversation, not just passively answer questions."},

        {"role": "user", "content": f"""
        ## Task Description
        Now enter the role-playing mode. In the following conversation, you will play as a Persuader who is trying to persuade the Persuadee to donate to the charity called Save the Children.

        ## Task Information
        Save the Children is head-quartered in London, and they work to help fight poverty around the world. Children need help in developing countries and war zones. Small donations like $1 or $2 go a long way to help.
        Your Goal: You are the Persuader who is trying to convince the Persuadee to donate to the charity called Save the Children.

        ## Dialogue Strategy Set
        {infos['strategy_set']}

        ## Dialogue History
        {conversation}
        
        You should follow the steps below to reason and generate a response:
        
        Step 1: State Analysis  
        Analyze the current dialogue progress. Then infer the mental states and predict future actions of the Persuadee.
        Do NOT mention, select, or hint at any specific strategy names in this step. Save the decision making for Step 2.
        Enclosed within <state_analysis> </state_anlysis> tags. 
        
        Step 2: Strategy Selection & Justification
        Based on your analysis, select the ONE most appropriate strategy from the "Dialogue Strategy Set" and provide a 'Brief Reason' to explain WHY you chose it.
        Enclosed within <strategy> </strategy> tags. Format: "[Strategy Name]:[Brief Reason]"
           
        Step 3: Response Generation
        Generate a concise, natural, and persuasive response that implements the strategy you selected.
        Enclosed within <response> </response> tags. 
        
        ## Output Format Requirements
        Output strictly in the following format and do not change the headers:
        <state_analysis>XXX</state_analysis><strategy>XXX</strategy><response>XXX</response>
        """}
    ]
    else:
        raise NotImplementedError

    return messages

def get_message_prompt_func_change(env, role, infos, conversation=None):
    """
    生成策略、思考与回复的联合Prompt构建函数
    """
    if env == "P4G":
        assert role == "Persuader", "Role must be 'Persuader' in P4G environment"

        messages = [
        {"role": "system", "content": "You are a dialogue expert skilled in using various dialogue strategies to accomplish tasks. You need to proactively and strategically guide the conversation, not just passively answer questions."},

        {"role": "user", "content": f"""
        ## Task Description
        Now enter the role-playing mode. In the following conversation, you will play as a Persuader who is trying to persuade the Persuadee to donate to the charity called Save the Children.

        ## Task Information
        Save the Children is head-quartered in London, and they work to help fight poverty around the world. Children need help in developing countries and war zones. Small donations like $1 or $2 go a long way to help.
        Your Goal: You are the Persuader who is trying to convince the Persuadee to donate to the charity called Save the Children.

        ## Dialogue Strategy Set
        {infos['strategy_set']}

        ## Dialogue History
        {conversation}
        
        You should follow the steps below to reason and generate a response:
        
        
        Step 1: Strategy Selection & Justification
        Based on your analysis, select the ONE most appropriate strategy from the "Dialogue Strategy Set" and provide a 'Brief Reason' to explain WHY you chose it
        Enclosed within <strategy> </strategy> tags. Format: "[Strategy Name]:[Brief Reason]"
           
        Step 2: Response Generation
        Generate a concise, natural, and persuasive response that implements the strategy you selected.
        Enclosed within <response> </response> tags. 
        
        Step 3: State Analysis
        Analyze the current dialogue progress. Then infer the mental states and predict future actions of the Persuadee.
        Enclosed within <state_analysis> </state_anlysis> tags.
        
        ## Output Format Requirements
        Output strictly in the following format and do not change the headers:
        <strategy>XXX</strategy><response>XXX</response><state_analysis>XXX</state_analysis>
        """}
    ]
    else:
        raise NotImplementedError

    return messages
# messages = [
#             {
#                 'role': 'system',
#                 'content': """
#                 You are a dialogue expert skilled in using various conversational strategies to accomplish tasks. You need to proactively and strategically guide the conversation, not just passively answer questions.
#             """
#             },
#             {
#                 'role': 'user',
#                 'content': f"""
#                 Now enter the role-playing mode. In the following conversation, you will play as a Persuader who is trying to persuade the Persuadee to donate to the charity called Save the Children.
                
#                 ## Task Information
#                 Save the Children is head-quartered in London, and they work to help fight poverty around the world. Children need help in developing countries and war zones. Small donations like $1 or $2 go a long way to help.
#                 Your Goal: You are the Persuader who is trying to convince the Persuadee to donate to the charity called Save the Children.
                
#                 ## Thinking Process
#                 1. In each turn, you must first analyze the current state based on the dialogue history.
#                 2. Then, select a strategy from the provided strategy set. 
#                 3. If you believe none of the strategies in the set are appropriate, you can generate your own response and summarize the strategy you used. **(Format: strategy_name: strategy_content, the content should be concise and accurate.)**
#                 3. Finally, generate your response based on the chosen strategy and adhere to the output format.
                                
#                 ## Dialogue Strategy Set
#                 {infos['strategy_set']}
                
#                 ## Dialogue History
#                 {conversation}
                
#                 ## Please strictly follow the format below for your output:
#                 ```json
#                 {{
#                     "strategy": "The strategy you have chosen eg. Greeting, Logical Appeal; Or the new strategy you generated (format: strategy_name: strategy_content, e.g., Flexible Contribution Reassurance: Emphasize that the donation amount can be freely chosen, so it will not impact personal finances.",
#                     "response": "Your response content, which should be a short and persuasive sentence.",
#                     "reason": "Briefly write down your thinking process here."
#                 }}
#                 ```
#                 """
#             }
#         ]


        # messages = [
        #     {"role": "system", "content": "You are a dialogue expert skilled in using various conversational strategies to accomplish tasks. You need to proactively and strategically guide the conversation, not just passively answer questions."},

        #     {"role": "user", "content": f"""
        #     ## Task Description
        #     Now enter the role-playing mode. In the following conversation, you will play as a Persuader who is trying to persuade the Persuadee to donate to the charity called Save the Children.

        #     ## Task Information
        #     Save the Children is head-quartered in London, and they work to help fight poverty around the world. Children need help in developing countries and war zones. Small donations like $1 or $2 go a long way to help.
        #     Your Goal: You are the Persuader who is trying to convince the Persuadee to donate to the charity called Save the Children.

        #     ## Dialogue Strategy Set
        #     {infos['strategy_set']}

        #     ## Dialogue History
        #     {conversation}

        #     ## Thinking Process
        #       1. In each turn, you must first analyze the current state . You should analyze the given dialogue history, especially the user's last response, to assess the current conversation progress, the user's mental state, and the action they took. 
        #       2. Then,  based on your analysis, decide on the best strategy for the next turn.
        #       3. Finally, generate your response based on the chosen strategy and the dialogue history.


        #     ## Output Format Requirements
        #     Please strictly adhere to the following format. Do not change the headers:

        #     **Thought Process**: Write your analysis here
        #     **Strategy**: The strategy you have chosen
        #     **Response**: Your response content, which should be a concise and persuasive sentence. 
        #     """}
        # ]



        # messages = [
        #     {"role": "system", "content": "You are a dialogue expert skilled in using various conversational strategies to accomplish tasks. You need to proactively and strategically guide the conversation, not just passively answer questions."},

        #     {"role": "user", "content": f"""
        #     ## Task Description
        #     Now enter the role-playing mode. In the following conversation, you will play as a Persuader who is trying to persuade the Persuadee to donate to the charity called Save the Children.

        #     ## Task Information
        #     Save the Children is head-quartered in London, and they work to help fight poverty around the world. Children need help in developing countries and war zones. Small donations like $1 or $2 go a long way to help.
        #     Your Goal: You are the Persuader who is trying to convince the Persuadee to donate to the charity called Save the Children.

        #     ## Dialogue Strategy Set
        #     {infos['strategy_set']}

        #     ## Dialogue History
        #     {conversation}
            
        #     ## Your Task
        #     You must strictly follow this three-step process to generate your output:

        #     1. contextual analysis:
        #     First, you should analyze the current dialogue progress. 
        #     Second, infer the mental states and future actions of the persuadee.
        #     Do not mention or select strategies in this phase.
        #     Enclosed within <state_analysis> </state_anlysis> tags.
            
        #     2. strategy decision:
        #     Based on your above analysis, you need to explain your thought process and rationale for choosing a strategy
        #     Finally conclude this section with the exact sentence: "The most appropriate strategy is [<<Strategy Name>>]".
        #     Enclose it in <strategy> </strategy> tags.

        #     3. response generation:
        #     Generate a concise, natural, and persuasive response that implements the strategy you selected.
        #     Enclose it within <response> </response> tags.
            
        #     ## Output Format Requirements
        #     Please strictly adhere to the following format. Do not change the headers or tags:
            
        #     <state_analysis> [Analyze the user's state and dialogue progress here]</state_analysis>
        #     <strategy>[Explain your strategy selection process here].The most appropriate strategy is [<<Strategy Name>>] </strategy>
        #     <response> [Write your generated response here] </response>
            
        #     """}
        # ]
