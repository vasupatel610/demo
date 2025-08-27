# app.py ── Streamlit UI for the FastAPI search service
import streamlit as st
import requests
import pandas as pd
from difflib import SequenceMatcher   # 🔑 for similarity check

# -------------------------------------------------------------------
# 1.  Configuration
# -------------------------------------------------------------------
API_URL = "http://localhost:8000"
HEALTH_EP = f"{API_URL}/health"
SEARCH_EP = f"{API_URL}/search"

# Initialize session state for the query
if 'query' not in st.session_state:
    st.session_state.query = ""

# Define the predefined questions
PREDEFINED_QUESTIONS = [
    "DKNY Unisex Trolley Bag",
    "Navy Blue Casual Shirt Under 1500",
    "Women Blue Printed Kurta",
    "Men Formal Slip-Ons",
    "Cropped Jeans for Women"
]

# -------------------------------------------------------------------
# 2.  Check back-end status once at start-up
# -------------------------------------------------------------------
@st.cache_data(ttl=60)   # refresh every 60 s
def get_backend_status():
    """Checks the health of the FastAPI back-end."""
    try:
        return requests.get(HEALTH_EP, timeout=5).json()
    except Exception as e:
        return {"status": "unreachable", "detail": str(e)}

status = get_backend_status()
st.sidebar.title("Back-end status")
st.sidebar.write(status)

if status.get("status") != "healthy":
    st.error("FastAPI back-end is not reachable — start it first.")
    st.stop()

# -------------------------------------------------------------------
# 3.  Page layout
# -------------------------------------------------------------------
st.title("🔎 Natural-language product search")

# Function to update the query when a button is clicked
def set_query(q):
    st.session_state.query = q

# Display predefined questions as buttons
st.write("Or try one of these:")
buttons_cols = st.columns(len(PREDEFINED_QUESTIONS))
for i, q in enumerate(PREDEFINED_QUESTIONS):
    with buttons_cols[i]:
        if st.button(q):
            set_query(q)

# Search input tied to session state
query = st.text_input(
    "Enter search query", 
    placeholder="e.g. blue denim jacket under ₹2,000",
    value=st.session_state.query,
    key="search_input"
)

# Other controls for search parameters
top_k = st.number_input(
    "Results", 
    min_value=1, 
    max_value=50, 
    value=10, 
    step=1
)
alpha = st.slider(
    "Semantic weight (α)", 
    0.0, 
    1.0, 
    value=status.get("embedding_method") == "groq" and 0.7 or 0.3, 
    step=0.05
)

# -------------------------------------------------------------------
# 4.  Call API and display results
# -------------------------------------------------------------------

def search_api(q, k, a):
    """Makes a POST request to the search API."""
    payload = {"query": q, "top_k": k, "alpha": a}
    r = requests.post(SEARCH_EP, json=payload, timeout=20)
    r.raise_for_status()
    return r.json()

def text_similarity(a: str, b: str) -> float:
    """Compute simple similarity between two strings."""
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()

import re
from collections import Counter

def rerank_results(hits, query):
    """Re-rank hits based on keyword overlap between query and product fields."""
    
    # Tokenize helper
    def tokenize(text):
        return re.findall(r"\w+", text.lower())
    
    query_tokens = tokenize(query)
    query_counter = Counter(query_tokens)

    for h in hits:
        combined_text = h.get("product_name", "") + " " + h.get("product_description", "")
        doc_tokens = tokenize(combined_text)
        doc_counter = Counter(doc_tokens)

        # Simple overlap score
        overlap = sum((query_counter & doc_counter).values())

        # Bonus: if query tokens appear in product_name, weight higher
        name_tokens = tokenize(h.get("product_name", ""))
        name_overlap = len(set(query_tokens) & set(name_tokens))

        h["local_score"] = overlap + 2 * name_overlap   # weights: name overlap more important

    # Sort descending by local_score
    return sorted(hits, key=lambda x: x["local_score"], reverse=True)

if query: # Only run a search if the query is not empty
    with st.spinner("Searching…"):
        try:
            res = search_api(query, int(top_k), float(alpha))
        except Exception as e:
            st.error(f"API call failed: {e}")
            st.stop()

    hits = res["results"]

    # 🔑 Re-rank using query vs product_name + description
    hits = rerank_results(hits, query)

    if not hits:
        st.info("No matches found.")
    else:
        st.success(f"{res['total_results']} results (method: {res['search_method']}, reranked locally)")

        for i, h in enumerate(hits):
            st.markdown("---")
            col1, col2 = st.columns([1, 4])

            with col1:
                st.metric("Local Similarity", f"{h['local_score']:.3f}")
                st.metric("Price", f"₹{h['price']:,.0f}")
                
            with col2:
                st.markdown(f"#### {h.get('highlighted_product_name', h['product_name'])}", unsafe_allow_html=True)
                st.markdown(h.get('highlighted_product_description', h['product_description']), unsafe_allow_html=True)
                
                if h.get("enhanced_fields"):
                    fields_str = " | ".join([f"**{k}**: {v}" for k, v in h["enhanced_fields"].items()])
                    st.markdown(f"**Details**: {fields_str}", unsafe_allow_html=True)
        
        with st.expander("Field analysis (from back-end)"):
            st.json(res.get("field_analysis", {}))


# # app.py ── Streamlit UI for the FastAPI search service
# import streamlit as st
# import requests
# import pandas as pd

# # -------------------------------------------------------------------
# # 1.  Configuration
# # -------------------------------------------------------------------
# API_URL = "http://localhost:8000"
# HEALTH_EP = f"{API_URL}/health"
# SEARCH_EP = f"{API_URL}/search"

# # Initialize session state for the query
# if 'query' not in st.session_state:
#     st.session_state.query = ""

# # Define the predefined questions
# PREDEFINED_QUESTIONS = [
#     "DKNY Unisex Trolley Bag",
#     "Navy Blue Casual Shirt Under 1500",
#     "Women Blue Printed Kurta",
#     "Men Formal Slip-Ons",
#     "Cropped Jeans for Women"
# ]

# # -------------------------------------------------------------------
# # 2.  Check back-end status once at start-up
# # -------------------------------------------------------------------
# @st.cache_data(ttl=60)   # refresh every 60 s
# def get_backend_status():
#     """Checks the health of the FastAPI back-end."""
#     try:
#         return requests.get(HEALTH_EP, timeout=5).json()
#     except Exception as e:
#         return {"status": "unreachable", "detail": str(e)}

# status = get_backend_status()
# st.sidebar.title("Back-end status")
# st.sidebar.write(status)

# if status.get("status") != "healthy":
#     st.error("FastAPI back-end is not reachable — start it first.")
#     st.stop()

# # -------------------------------------------------------------------
# # 3.  Page layout
# # -------------------------------------------------------------------
# st.title("🔎 Natural-language product search")

# # Function to update the query when a button is clicked
# def set_query(q):
#     st.session_state.query = q

# # Display predefined questions as buttons
# st.write("Or try one of these:")
# buttons_cols = st.columns(len(PREDEFINED_QUESTIONS))
# for i, q in enumerate(PREDEFINED_QUESTIONS):
#     with buttons_cols[i]:
#         if st.button(q):
#             set_query(q)

# # This is the key change: we've removed the `with st.form` block.
# # The search input is now directly tied to the session state.
# query = st.text_input(
#     "Enter search query", 
#     placeholder="e.g. blue denim jacket under ₹2,000",
#     value=st.session_state.query,
#     key="search_input"
# )

# # Other controls for search parameters
# top_k = st.number_input(
#     "Results", 
#     min_value=1, 
#     max_value=50, 
#     value=10, 
#     step=1
# )
# alpha = st.slider(
#     "Semantic weight (α)", 
#     0.0, 
#     1.0, 
#     value=status.get("embedding_method") == "groq" and 0.7 or 0.3, 
#     step=0.05
# )

# # -------------------------------------------------------------------
# # 4.  Call API and display results
# # -------------------------------------------------------------------

# def search_api(q, k, a):
#     """Makes a POST request to the search API."""
#     payload = {"query": q, "top_k": k, "alpha": a}
#     r = requests.post(SEARCH_EP, json=payload, timeout=20)
#     r.raise_for_status()
#     return r.json()

# if query: # Only run a search if the query is not empty
#     with st.spinner("Searching…"):
#         try:
#             res = search_api(query, int(top_k), float(alpha))
#         except Exception as e:
#             st.error(f"API call failed: {e}")
#             st.stop()

#     hits = res["results"]
#     if not hits:
#         st.info("No matches found.")
#     else:
#         st.success(f"{res['total_results']} results (method: {res['search_method']})")

#         for i, h in enumerate(hits):
#             st.markdown("---")
#             col1, col2 = st.columns([1, 4])

#             with col1:
#                 st.metric("Score", f"{h['score']:.3f}")
#                 st.metric("Price", f"₹{h['price']:,.0f}")
                
#             with col2:
#                 # Use st.markdown with unsafe_allow_html=True to render the HTML tags
#                 st.markdown(f"#### {h.get('highlighted_product_name', h['product_name'])}", unsafe_allow_html=True)
#                 st.markdown(h.get('highlighted_product_description', h['product_description']), unsafe_allow_html=True)
                
#                 if h.get("enhanced_fields"):
#                     fields_str = " | ".join([f"**{k}**: {v}" for k, v in h["enhanced_fields"].items()])
#                     st.markdown(f"**Details**: {fields_str}", unsafe_allow_html=True)
        
#         with st.expander("Field analysis (from back-end)"):
#             st.json(res.get("field_analysis", {}))


# # app.py ── Streamlit UI for the FastAPI search service
# import streamlit as st
# import requests
# import pandas as pd

# # -------------------------------------------------------------------
# # 1.  Configuration
# # -------------------------------------------------------------------
# API_URL   = st.secrets.get("api_url", "http://localhost:8000")
# HEALTH_EP = f"{API_URL}/health"
# SEARCH_EP = f"{API_URL}/search"

# # -------------------------------------------------------------------
# # 2.  Check back-end status once at start-up
# # -------------------------------------------------------------------
# @st.cache_data(ttl=60)   # refresh every 60 s
# def get_backend_status():
#     try:
#         return requests.get(HEALTH_EP, timeout=5).json()
#     except Exception as e:
#         return {"status": "unreachable", "detail": str(e)}

# status = get_backend_status()
# st.sidebar.title("Back-end status")
# st.sidebar.write(status)

# if status.get("status") != "healthy":
#     st.error("FastAPI back-end is not reachable — start it first.")
#     st.stop()

# # -------------------------------------------------------------------
# # 3.  Page layout
# # -------------------------------------------------------------------
# st.title("🔎 Smart Product Search by Artisans Commerce Cloud")

# with st.form(key="query_form"):
#     query   = st.text_input("Enter search query", placeholder="e.g. blue denim jacket under ₹2,000")
#     top_k   = st.number_input("Results", 1, 50, 10, step=1)
#     alpha   = st.slider("Semantic weight (α)", 0.0, 1.0, status.get("embedding_method") == "groq" and 0.7 or 0.3, 0.05)
#     submitted = st.form_submit_button("Search")

# # -------------------------------------------------------------------
# # 4.  Call API and display results
# # -------------------------------------------------------------------
# def search_api(q, k, a):
#     payload = {"query": q, "top_k": k, "alpha": a}
#     r = requests.post(SEARCH_EP, json=payload, timeout=20)
#     r.raise_for_status()
#     return r.json()

# if submitted:
#     if not query.strip():
#         st.warning("Please type a query.")
#         st.stop()

#     with st.spinner("Searching…"):
#         try:
#             res = search_api(query, int(top_k), float(alpha))
#         except Exception as e:
#             st.error(f"API call failed: {e}")
#             st.stop()

#     hits = res["results"]
#     if not hits:
#         st.info("No matches found.")
#     else:
#         st.success(f"{res['total_results']} results (method: {res['search_method']})")
#         # Build a DataFrame for pretty display
#         records = []
#         for h in hits:
#             record = {
#                 "Score": f"{h['score']:.3f}",
#                 "Name":  h["product_name"],
#                 "Description": h["product_description"][:80] + "…" if len(h["product_description"]) > 80 else h["product_description"],
#                 "Price": f"₹{h['price']:,.0f}",
#             }
#             # flatten enhanced fields into their own columns
#             if h.get("enhanced_fields"):
#                 record.update(h["enhanced_fields"])
#             records.append(record)

#         st.dataframe(pd.DataFrame(records))

#         with st.expander("Field analysis (from back-end)"):
#             st.json(res.get("field_analysis", {}))
