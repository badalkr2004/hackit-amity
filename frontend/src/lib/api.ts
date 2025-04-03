export const getSentiment = async (text: string): Promise<{
    sentiment: string
}> => {
    console.log()
    const resp = await fetch("http://localhost:8000/predict", {
        method: "POST",
        body: JSON.stringify({ text })
    });

    if (!resp.ok) {
        throw new Error(`Error: ${resp.status} ${resp.statusText}`);
    }

    const data = await resp.json();
    
    // Validate the response structure
    if (data && typeof data.label === 'string') {
        return {
            sentiment: data.label
        };
    } else {
        throw new Error("Invalid response structure");
    }
}

