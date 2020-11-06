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
                    Your sentence is being processed. The status will auto-refresh every 5 seconds.
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
        this.state = {
            squashId: squashId,
            settings: {
                'top_p_paraphrase': 0.0,
                'top_p_style': 0.6,
            },
            ans_mode: 'original',
            forest: null,
            queue_number: null,
            input_text: null,
            status: null,
            dropdownOpen: false
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
        this.interval = setInterval(this.getSquashedDocument.bind(this), 5000);
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

    transferSentence() {
        var url = SERVER_URL + "/request_strap_doc";
        var flags = {
            method: 'POST',
            body: JSON.stringify({
                settings: this.state.settings,
                input_text: document.getElementById('strapInputText').value
            })
        };
        fetch(url, flags).then(res => res.json()).then((result) => {
            window.location.href = '/?id=' + result.new_id;
        }, (error) => {
            console.log(error);
        })
    }

    render() {
        var squash_loaded = false;
        if (this.state.forest != null) {
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
                        <p>This system rewrites text using a specified target style while preserving semantic information. No parallel data was used to train this system.
                         Check out our <a href="http://style.cs.umass.edu/">landing page</a> for more details.
                         Feel free to fork and use the <a href="https://github.com/martiansideofthemoon/style-transfer-paraphrase/tree/master/web-demo">source code</a> for this demo.</p>
                        <p>Contact <a href="mailto:kalpesh@cs.umass.edu">kalpesh@cs.umass.edu</a> if you run into any issues.</p>

                    </Col>
                    <Col md={{order: 2, size: 7}} xs={{order: 2}}>
                    </Col>
                </Row>
                <Row>
                    <Col md={{order: 2, size: 5}} xs={{order: 2}}>
                    <div>
                    <div className="precomputed-div">
                    Enter your text or&nbsp;&nbsp;
                    <Dropdown isOpen={this.state.dropdownOpen} toggle={() => this.toggleDropDown()}>
                        <DropdownToggle color="info" caret>
                          choose an example
                        </DropdownToggle>
                        <DropdownMenu>
                          <DropdownItem tag="a" href="/?id=04bf8a42f934944809e76ec1">Cricket</DropdownItem>
                          <DropdownItem tag="a" href="/?id=bbd2ba3e440ff9abb52f211f">The Fellowship of the Ring</DropdownItem>
                          <DropdownItem tag="a" href="/?id=e3aebbb37fa9d1c48638aa46">Neil Armstrong</DropdownItem>
                        </DropdownMenu>
                    </Dropdown>
                    </div>
                    <hr />

                    <RequestForm
                        settings={this.state.settings}
                        changeSliderParaphrase={(e) => this.changeSlider(e, 'top_p_paraphrase')}
                        changeSliderStyle={(e) => this.changeSlider(e, 'top_p_style')}
                        transferSentence={() => this.transferSentence()}
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
