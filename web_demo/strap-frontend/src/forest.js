import React from 'react';
import {
    Badge,
    Container,
    Col,
    Button,
    Row
} from 'reactstrap';

import Toggle from 'react-toggle'

function chooseAnswerType(qa_object, ans_mode) {
    // For SPECIFIC questions generated from sentences, display the predicted answer as far as possible
    // Do this even when the ans_mode is original
    if (ans_mode === 'original' && qa_object.algorithm !== 'specific_sent') {
        return qa_object.answers[0].text;
    } else if (qa_object.predicted_answer) {
        return qa_object.predicted_answer;
    } else {
        return qa_object.answers[0].text;
    }
}

function SpecificQA(props) {
    return (
        <Row>
            <Col xs="2"></Col>
            <Col xs="10">
                <div className='specific-qa-view'>
                    <p className='specific-qa-text'>
                        <span className='specific-question'><b>{'specific-question: ' + props.specific_qa.question}</b></span><br/>
                        <span className='specific-answer'>{'specific-answer: ' + chooseAnswerType(props.specific_qa, props.ans_mode)}</span>
                    </p>
                </div>
            </Col>
        </Row>
    );
}

function QATree(props) {

    const specific_list = props.qa_tree.children.map((specific_qa, index) => {
        return <SpecificQA specific_qa={specific_qa} key={specific_qa.id} ans_mode={props.ans_mode} />
    })

    return (
        <div>
        <Row>
            <Col xs="1" className="general-button-div">
                    <Button color="info" onClick={props.toggle} disabled={props.qa_tree.children.length === 0}>{props.expanded ? '-' : '+'}</Button>
            </Col>
            <Col xs="11">
                <div className='general-qa-view'>
                    <p className='general-qa-text'>
                        <span className='general-question'><b>{'general-question: ' + props.qa_tree.question}</b></span><br/>
                        <span className='general-answer'>{'general-answer: ' + chooseAnswerType(props.qa_tree, props.ans_mode)}</span>
                    </p>
                </div>
            </Col>
        </Row>
        {props.expanded && specific_list}
        </div>
    );
}

function ParagraphForest(props) {
    const qa_list = props.para_forest.binned_qas.map((qa_tree, qa_index) => {
        return <QATree
                qa_tree={qa_tree}
                key={qa_tree.id}
                ans_mode={props.ans_mode}
                toggle={() => props.toggle(qa_index)}
                expanded={props.expanded[qa_index]}/>
    })
    return (
        <div className='para-view'>
            <h4>Paragraph #{props.number}</h4>
            <div className='para-text-div'>
                <p className='para-text'>{props.para_forest.para_text}</p>
            </div>
            <div className="para-forest">{qa_list}</div>
            <br />
        </div>
    );
}

function SquashForest(props) {
    const para_list = props.forest.qa_tree.map((para, para_index) => {
        return <ParagraphForest
                para_forest={para}
                key={'para_num_' + para_index}
                number={para_index + 1}
                ans_mode={props.ans_mode}
                toggle={(qa_index) => props.toggleSpecific(para_index, qa_index)}
                expanded={props.expanded[para_index]}/>
    })
    return (
        <div>
        <p>Press the + button to expand a question tree. </p>
        <div className="squash-forest">{para_list}</div>
        </div>
    );
}

export default SquashForest;