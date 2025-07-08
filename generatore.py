def generate_descriptions(row, api_key):
    client = OpenAI(api_key=api_key)

    product_info = ", ".join([
        f"{k}: {v}" for k, v in row.items()
        if k.lower() not in ["description", "short_description"] and pd.notna(v)
    ])

    prompt = f"""
Scrivi due descrizioni per una calzatura:
1. Descrizione lunga di circa 60 parole.
2. Descrizione breve di circa 20 parole.
Tono: accattivante, caldo, professionale, user-friendly e SEO-friendly.
Dettagli: {product_info}
    """

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
    )

    result = response.choices[0].message.content.strip()
    parts = [p.strip("1234567890.-: \n") for p in result.split("\n") if p.strip()]
    long_desc = parts[0] if len(parts) > 0 else "Descrizione non trovata"
    short_desc = parts[1] if len(parts) > 1 else long_desc[:100]

    return long_desc, short_desc
