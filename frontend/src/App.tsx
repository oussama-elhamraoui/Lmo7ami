
import './App.css'
import axios from 'axios';

function App() {

    axios.post(
      "http://127.0.0.1:8000",
    ).then((response) => {
      console.log(response.data);
    })
    

  return (
    <>
      
    </>
  )
}

export default App
