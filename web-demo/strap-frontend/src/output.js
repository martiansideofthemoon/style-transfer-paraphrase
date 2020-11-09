import React from 'react';

function StyleOutput(props) {
    var missingMessage = "Please enter a target style"
    return (
        <div>
        <b>Input</b>: {props.output_data.input_text}
        <br /><br />

        <b>Sample #1</b>
        <br />
        <b>Intermediate Paraphrase</b>: {props.output_data.paraphrase[0]}<br />
        <b>Style Transferred Output ({props.output_data.target_style ? props.output_data.target_style : missingMessage})</b>: {props.output_data.style_transfer[0]}
        <br /><br />

        <b>Sample #2</b>
        <br />
        <b>Intermediate Paraphrase</b>: {props.output_data.paraphrase[1]}<br />
        <b>Style Transferred Output ({props.output_data.target_style ? props.output_data.target_style : missingMessage})</b>: {props.output_data.style_transfer[1]}
        <br /><br />

        <b>Sample #3</b>
        <br />
        <b>Intermediate Paraphrase</b>: {props.output_data.paraphrase[2]}<br />
        <b>Style Transferred Output ({props.output_data.target_style ? props.output_data.target_style : missingMessage})</b>: {props.output_data.style_transfer[2]}
        <br /><br />

        <b>Sample #4</b>
        <br />
        <b>Intermediate Paraphrase</b>: {props.output_data.paraphrase[3]}<br />
        <b>Style Transferred Output ({props.output_data.target_style ? props.output_data.target_style : missingMessage})</b>: {props.output_data.style_transfer[3]}
        <br /><br />

        <b>Sample #5</b>
        <br />
        <b>Intermediate Paraphrase</b>: {props.output_data.paraphrase[4]}<br />
        <b>Style Transferred Output ({props.output_data.target_style ? props.output_data.target_style : missingMessage})</b>: {props.output_data.style_transfer[4]}
        <br /><br />
        </div>
    );
}

export default StyleOutput;
