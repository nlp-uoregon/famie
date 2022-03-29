import React from 'react';
import { makeStyles } from '@material-ui/core/styles';
import Table from '@material-ui/core/Table';
import TableBody from '@material-ui/core/TableBody';
import TableCell from '@material-ui/core/TableCell';
import TableContainer from '@material-ui/core/TableContainer';
import TableHead from '@material-ui/core/TableHead';
import TableRow from '@material-ui/core/TableRow';
import Paper from '@material-ui/core/Paper';
import DeleteIcon from '@material-ui/icons/Delete';
import IconButton from '@material-ui/core/IconButton';
import TouchAppIcon from '@material-ui/icons/TouchApp';
import Tooltip from '@material-ui/core/Tooltip';
import HelpIcon from '@material-ui/icons/Help';


const useStyles = makeStyles({
    tablecell: {
        fontSize: '80%'
    }
  });

const StyledTableCell = (props) => {
    const classes = useStyles();
    return (
        <TableCell 
            align={props.align}
            className={classes.tablecell}
            component={props.component}
            scope={props.scope}
        >
        {props.content}
        </TableCell>
    )
}

const RuleTableHead = (props) => (
    <TableHead>
        <TableRow className={props.classes.table}>
            <StyledTableCell content={"Rule name"}/>
            <StyledTableCell align="right" content="Coverage"/>
            <StyledTableCell align="right" content="Overlaps"/>
            <StyledTableCell align="right" content="Conflicts"/>
            <StyledTableCell align="right" content="Accuracy"/>
            <StyledTableCell align="right" content="Class"/>
        </TableRow>
    </TableHead>
)

const TotalRuleTableRow = (props) => {
    const row = {'coverage': props.totalRulesCoverage || 0, 
                'overlaps': props.totalRulesOverlaps, 
                'conflicts': props.totalRulesConflicts, 
                'name': 'Covered by rules'}
    return (
        <RuleTableRow 
            key={"total_rules"}
            row={row}
            actionExplore
            exploreRule={props.exploreRule}
            index={-1}
            classes={props.classes}
        />
    )
}

const UnlabelledRuleTableRow = (props) => {

    // not covered by merged rule or manual labels
    const coverage = props.totalDocuments - props.totalRulesCoverage - props.totalManualDocs +  props.totalDocsRulesManualLabelled;

    const row = {'coverage': coverage, 
                'name': 'Not covered by rules or manual labels'}
    return (
        <RuleTableRow 
            row={row}
            actionExplore
            exploreRule={props.exploreRule}
            index={-2}
            classes={props.classes}
        />
    )
}

const TotalMergedTableRow = (props) => {
    const mergedMethod = getMergedMethod(props.mergedMethod);
    console.log(`Merged method is ${mergedMethod}`);

    const row = {'coverage': props.totalPredictedDocs || 0, 
                'accuracy': `${props.totalCorrect}/${props.totalCorrect + props.totalIncorrect}`, 
                'name': mergedMethod? `Total merged label ${mergedMethod}` : "Total merged label"}
    return (
        <RuleTableRow 
            key={"total_merged"}
            row={row}
            outcome={""}
            classes={props.classes}
        />
    )
}

const RuleName = (props) => {
    return (
        <div style={{display: 'flex', 
        alignItems: 'center', 
        justifyContent: 'flex-start'}}>
            {props.name}
            {props.explainRule &&
                <Tooltip title={props.explanation} style={{marginLeft: '10px'}}>
                    <HelpIcon/>
                </Tooltip>
            }
        </div>)
}

const RuleTableRow = (props) => {
    return (
        <TableRow key={props.row.name} className={props.classes.table}>
            <StyledTableCell
                component="th" 
                scope="row"
                content={<RuleName 
                            name={props.row.name}
                            explanation={props.explanation}
                            explainRule={props.explainRule}
                        />}
            />
            <StyledTableCell align="right" content={props.row.coverage}/>
            <StyledTableCell align="right" content={props.row.overlaps}/>
            <StyledTableCell align="right" content={props.row.conflicts}/>
            <StyledTableCell align="right" content={props.row.accuracy}/>
            <StyledTableCell align="right" content={props.outcome}/>
            <StyledTableCell align="right" 
                            content={props.actionDelete &&  
                                        <RemoveButton 
                                            ruleIndex={props.index}
                                            deleteRule={props.deleteRule}
                                            disableDeleting={props.disableDeleting}
                                        />}
            />
            <StyledTableCell align="right" 
                            content={props.actionExplore &&  
                                        <ExploreButton 
                                            ruleIndex={props.index}
                                            exploreRule={props.exploreRule}
                                        />}
            />
        </TableRow>
    )
}

const RemoveButton = (props) => (
    <Tooltip title="Delete">
        <IconButton 
            aria-label="remove" 
            onClick={(e) => props.deleteRule(e, props.ruleIndex)} 
            name="remove-button"
            disabled={props.disableDeleting}
        >
            <DeleteIcon name="remove-icon"/>
        </IconButton>
    </Tooltip>
)

const ExploreButton = (props) => (
    <Tooltip title="Label">
        <IconButton aria-label="explore" onClick={(e) => props.exploreRule(e, props.ruleIndex)} name="explore-button">
            <TouchAppIcon name="explore-icon"/>
        </IconButton>
    </Tooltip>
)

const getMergedMethod = (mergedMethod) => {
    switch(mergedMethod){
        case "majority": 
            return "(majority vote)";
        case "model":
            return "(model)";
    }
}

const RuleTableClassification = (props) => {

    return (
        <div>
            { props.rules && 
                <TableContainer component={Paper}>
                    <Table aria-label="project table">
                        <RuleTableHead classes={props.classes}/>
                        <TableBody>
                            {props.rules.map((row, index) => { 
                                const params = JSON.parse(row.params);
                                console.log('params', params);
                                console.log(props);
                                return (
                                    <RuleTableRow 
                                        key={`${row.name}-${index}`}
                                        explanation={params.rule_explanation}
                                        index={row.id}
                                        row={row}
                                        outcome={props.docs.classes[row.classId].name}
                                        deleteRule={props.deleteRule}
                                        exploreRule={props.exploreRule}
                                        disableDeleting={props.disableDeleting}
                                        classes={props.classes}
                                        actionDelete
                                        actionExplore
                                        explainRule
                                    />
                                )})}
                            <TotalRuleTableRow 
                                {...props.docs}
                                classes={props.classes}
                                exploreRule={props.exploreRule}
                            />
                            <UnlabelledRuleTableRow 
                                {...props.docs}
                                classes={props.classes}
                                exploreRule={props.exploreRule}
                            />
                            <TotalMergedTableRow 
                                {...props.docs}
                                classes={props.classes}
                            />
                        </TableBody>
                    </Table>
                </TableContainer>
            }
        </div>              
    )
}

export default RuleTableClassification;