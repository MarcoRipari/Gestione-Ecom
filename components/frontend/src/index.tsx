import React from "react"
import { Streamlit, StreamlitComponentBase } from "streamlit-component-lib"

interface Props {
    url: string
}

export default class GoogleOAuthPopup extends StreamlitComponentBase<Props> {

    openPopup() {
        const width = 500
        const height = 600
        const left = window.screenX + (window.innerWidth - width) / 2
        const top = window.screenY + (window.innerHeight - height) / 2
        const url = this.props.url

        const popup = window.open(
            url,
            "GoogleOAuth",
            `width=${width},height=${height},left=${left},top=${top}`
        )

        if (!popup) {
            alert("Impossibile aprire il popup. Controlla il blocco popup del browser.")
            return
        }

        // Ascolta messaggi dal popup
        const listener = (event: MessageEvent) => {
            if (event.data.type === "OAUTH_TOKEN") {
                Streamlit.setComponentValue(event.data.token)
                window.removeEventListener("message", listener)
                popup.close()
            }
        }
        window.addEventListener("message", listener)
    }

    render() {
        return (
            <button
                style={{
                    padding: "10px 20px",
                    fontSize: "16px",
                    borderRadius: "8px",
                    backgroundColor: "#4f46e5",
                    color: "white",
                    border: "none",
                    cursor: "pointer"
                }}
                onClick={() => this.openPopup()}
            >
                Login con Google
            </button>
        )
    }
}
