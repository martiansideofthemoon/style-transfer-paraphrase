import React from 'react';
import {
    Badge,
    Container,
    Col,
    Row,
    Button,
    Form,
    Input,
    Card,
    Table,
    FormText,
    Dropdown, DropdownToggle, DropdownMenu, DropdownItem,
} from 'reactstrap';
import StyleOutput from './output.js';
import RequestForm from './request_form.js';
import SERVER_URL from './url.js';
import ReactGA from 'react-ga';
import { Helmet } from 'react-helmet'
import Toggle from 'react-toggle'


ReactGA.initialize('UA-144025713-1');


function QueueNumber(props) {
    if (props.queue_number === 1) {
        return (
            <div>
                <div>
                    Your sentence is being processed. The status will auto-refresh every two seconds.
                </div>
            </div>
        )
    } else {
        return (
            <div>
                <div>
                    Your sentence is in the queue, {props.queue_number - 1} sentence(s) before you. The status will auto-refresh every 5 seconds.
                </div>
            </div>
        )
    }
}

class SquashDemo extends React.Component {
    constructor(props) {
        super(props);
        const urlParams = new URLSearchParams(window.location.search);
        const squashId = urlParams.get('id');
        const initRandom = urlParams.get('random') === 'true';
        this.state = {
            squashId: squashId,
            settings: {
                'top_p_paraphrase': 0.0,
                'top_p_style': 0.7,
            },
            ans_mode: 'original',
            output_data: null,
            queue_number: null,
            input_text: null,
            status: null,
            dropdownOpen: false,
            styleDropDownOpen: false,
            targetStyle: null,
            initRandom: initRandom
        };
    }

    getSquashedDocument() {
        if (this.state.squashId && this.state.queue_number !== 0) {
            var url = SERVER_URL + "/get_strap_doc?id=" + this.state.squashId

            fetch(url).then(res => res.json()).then((result) => {
                if (result.input_text) {
                    document.getElementById("strapInputText").value = result.input_text
                }
                this.setState({
                    output_data: result.output_data,
                    queue_number: result.queue_number,
                    targetStyle: this.state.initRandom ? null : result.target_style,
                    settings: {
                        'top_p_paraphrase': result.settings.top_p_paraphrase,
                        'top_p_style': result.settings.top_p_style
                    },
                    status: result.status
                });

            }, (error) => {
                console.log(error)
            })
        }
    }

    componentDidMount() {
        this.getSquashedDocument.bind(this)();
        ReactGA.pageview(window.location.pathname + window.location.search);
        this.interval = setInterval(this.getSquashedDocument.bind(this), 2000);
    }

    componentWillUnmount() {
        clearInterval(this.interval);
    }

    changeSlider(e, type) {
        var new_settings = this.state.settings;
        new_settings[type] = e.target.value;
        this.setState({settings: new_settings});
    }

    toggleSpecific(para_index, qa_index) {
        var new_expanded = this.state.expanded;
        new_expanded[para_index][qa_index] = !this.state.expanded[para_index][qa_index]
        this.setState({
            expanded: new_expanded
        });
    }

    toggleAnswerMode() {
        this.setState({
            ans_mode: (this.state.ans_mode === 'original' ? 'predicted' : 'original')
        });
    }

      toggleDropDown() {
        this.setState(prevState => ({
          dropdownOpen: !prevState.dropdownOpen
        }));
      }

      toggleStyleDropDown() {
        this.setState(prevState => ({
          styleDropDownOpen: !prevState.styleDropDownOpen
        }));
      }

      toggleStyle(targetStyle) {
        this.setState({
            targetStyle: targetStyle
        });
      }

    transferSentence(random_sentence = false) {
        var url = SERVER_URL + "/request_strap_doc";
        var flags = {
            method: 'POST',
            body: JSON.stringify({
                settings: this.state.settings,
                input_text: document.getElementById('strapInputText').value,
                random: random_sentence,
                target_style: this.state.targetStyle
            })
        };
        fetch(url, flags).then(res => res.json()).then((result) => {
            window.location.href = '/?id=' + result.new_id + '&random=' + random_sentence;
        }, (error) => {
            console.log(error);
        })
    }

    render() {
        var squash_loaded = false;
        if (this.state.output_data != null) {
            squash_loaded = true;
        }
        return (
            <div className="container-fluid">
                <Helmet>
                    <meta charSet="utf-8" />
                    <title>Reformulating Unsupervised Style Transfer as Paraphrase Generation</title>
                </Helmet>
                <Row>
                    <Col md={{order: 2, size: 5}} xs={{order: 1}}>
                        <h5>A demo for <a href="https://arxiv.org/abs/2010.05700">Reformulating Unsupervised Style Transfer as Paraphrase Generation</a></h5>
                        <p>This system rewrites text using a specified target style while preserving semantic information. <br/> <b>No parallel style transfer data was used to train this system</b>.
                         Check out our <a href="http://style.cs.umass.edu/">landing page</a> for links to the code, paper and dataset. Note that we are not performing any filtering on our model outputs
                         and they might occasionally be biased like many <a href="https://arxiv.org/pdf/2005.14165.pdf#page=36">modern text generation systems</a>. <br />

                         The source code for the demo can be found <a href="https://github.com/martiansideofthemoon/style-transfer-paraphrase/tree/master/web-demo">here</a>.
                         Contact <a href="mailto:kalpesh@cs.umass.edu">kalpesh@cs.umass.edu</a> if you run into any issues.</p>
                    </Col>
                    <Col md={{order: 2, size: 7}} xs={{order: 2}}>
                    </Col>
                </Row>
                <Row>
                    <Col md={{order: 2, size: 5}} xs={{order: 2}}>
                    <div>
                    <hr />
                    <RequestForm
                        settings={this.state.settings}
                        changeSliderParaphrase={(e) => this.changeSlider(e, 'top_p_paraphrase')}
                        changeSliderStyle={(e) => this.changeSlider(e, 'top_p_style')}
                        transferSentence={() => this.transferSentence()}
                        styleDropDownOpen={this.state.styleDropDownOpen}
                        toggleStyleDropDown={() => this.toggleStyleDropDown()}
                        targetStyle={this.state.targetStyle}
                        toggleStyle={(e) => this.toggleStyle(e)}
                        toggleExamplesDropDown={() => this.toggleDropDown()}
                        transferSentenceRandom={() => this.transferSentence(true)}
                        dropdownOpen={this.state.dropdownOpen}
                    />
                    </div>
                    </Col>
                    <Col md={{order: 2, size: 7}} xs={{order: 1}}>
                        {squash_loaded && <StyleOutput output_data={this.state.output_data}/>}
                        {this.state.queue_number !== null && this.state.queue_number !== 0 && <QueueNumber queue_number={this.state.queue_number} status={this.state.status}/>}
                    </Col>
                </Row>
            </div>
        );
    }
}

export default SquashDemo;
